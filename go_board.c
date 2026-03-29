/*
 * go_board.c - High-performance Go board for MCTS
 *
 * Key data structure: Union-Find (DSU) for groups
 *
 * Instead of BFS to find groups/liberties (O(n) per check),
 * we maintain:
 *   - group_id[pos]:     which group this stone belongs to
 *   - liberty_count[gid]: how many liberties group gid has
 *   - group_color[gid]:   color of stones in group gid
 *
 * is_legal_move() is now O(4) (constant neighbors check)
 * play_move() is O(k) where k is number of captured stones
 * get_legal_moves() is O(361 * 4) instead of O(361 * BFS)
 *
 * Expected speedup: 10-50x for MCTS simulations
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX_SIZE 19
#define MAX_CELLS (MAX_SIZE * MAX_SIZE)

/* ─── Board state ─────────────────────────────────────────────────────────── */

typedef struct {
    PyObject_HEAD

    int size;
    int8_t  board[MAX_CELLS];       /* 0=empty, 1=black, -1=white          */

    /* Union-Find for groups */
    int     parent[MAX_CELLS];      /* DSU parent                           */
    int     rank[MAX_CELLS];        /* DSU rank for union by rank           */
    int     liberty_count[MAX_CELLS]; /* liberties of the GROUP rooted here */
    int8_t  group_color[MAX_CELLS]; /* color of group rooted here           */

    /* Ko */
    int     ko_point;               /* flat index, -1 = no ko              */

    /* Captures */
    int     black_captures;
    int     white_captures;

    /* Zobrist */
    uint64_t zobrist_table[2][MAX_CELLS];  /* [color_idx][pos]              */
    uint64_t zobrist_hash;

    /* Move info */
    int     last_move;              /* flat index, -1 = pass               */
    int     consecutive_passes;

} GoBoardObject;

/* ─── Index helpers ───────────────────────────────────────────────────────── */

static inline int idx(int size, int r, int c) { return r * size + c; }
static inline int in_bounds(int size, int r, int c) {
    return r >= 0 && r < size && c >= 0 && c < size;
}

static const int DR[4] = {-1, 1,  0, 0};
static const int DC[4] = { 0, 0, -1, 1};

/* ─── Union-Find ──────────────────────────────────────────────────────────── */

static int dsu_find(GoBoardObject *b, int x) {
    /* Path compression */
    while (b->parent[x] != x) {
        b->parent[x] = b->parent[b->parent[x]];  /* path halving */
        x = b->parent[x];
    }
    return x;
}

/*
 * Merge group of stone at `b` into group of stone at `a`.
 * Returns new root. Liberty counts are merged; shared liberties
 * (the point where the two groups now touch) must be subtracted
 * by the caller.
 */
static int dsu_union(GoBoardObject *b, int a, int c) {
    int ra = dsu_find(b, a);
    int rc = dsu_find(b, c);
    if (ra == rc) return ra;

    /* Union by rank */
    if (b->rank[ra] < b->rank[rc]) { int tmp = ra; ra = rc; rc = tmp; }
    b->parent[rc] = ra;
    if (b->rank[ra] == b->rank[rc]) b->rank[ra]++;

    /* Merge liberty counts */
    b->liberty_count[ra] += b->liberty_count[rc];

    return ra;
}

/* ─── Liberty accounting ──────────────────────────────────────────────────── */

/*
 * Count distinct liberties of the group containing `pos`.
 * Used for incremental repair only (slow path, called rarely).
 * Returns exact liberty count.
 */
static int count_group_liberties(GoBoardObject *b, int root) {
    /* BFS over all stones in the group, count unique empty neighbors */
    int size = b->size;
    int8_t visited_lib[MAX_CELLS];
    memset(visited_lib, 0, sizeof(visited_lib));

    int stack[MAX_CELLS];
    int8_t in_stack[MAX_CELLS];
    memset(in_stack, 0, sizeof(in_stack));

    int sp = 0, liberties = 0;

    /* Collect all stones in group */
    for (int pos = 0; pos < size * size; pos++) {
        if (b->board[pos] != 0 && dsu_find(b, pos) == root) {
            if (!in_stack[pos]) {
                stack[sp++] = pos;
                in_stack[pos] = 1;
            }
        }
    }

    for (int i = 0; i < sp; i++) {
        int pos = stack[i];
        int r = pos / size, c = pos % size;
        for (int d = 0; d < 4; d++) {
            int nr = r + DR[d], nc = c + DC[d];
            if (!in_bounds(size, nr, nc)) continue;
            int npos = idx(size, nr, nc);
            if (b->board[npos] == 0 && !visited_lib[npos]) {
                visited_lib[npos] = 1;
                liberties++;
            }
        }
    }
    return liberties;
}

/* ─── Initialization ──────────────────────────────────────────────────────── */

static void init_zobrist(GoBoardObject *b) {
    /* Simple xorshift64 PRNG with fixed seed for reproducibility */
    uint64_t state = 0xDEADBEEFCAFEBABEULL;
    for (int col = 0; col < 2; col++) {
        for (int pos = 0; pos < MAX_CELLS; pos++) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            b->zobrist_table[col][pos] = state;
        }
    }
}

static void reset_board(GoBoardObject *b) {
    int n = b->size * b->size;
    memset(b->board, 0, sizeof(b->board));
    for (int i = 0; i < n; i++) {
        b->parent[i] = i;
        b->rank[i] = 0;
        b->liberty_count[i] = 0;
        b->group_color[i] = 0;
    }
    b->ko_point          = -1;
    b->black_captures    = 0;
    b->white_captures    = 0;
    b->zobrist_hash      = 0;
    b->last_move         = -1;
    b->consecutive_passes = 0;
}

/* ─── is_legal (core O(4) check) ─────────────────────────────────────────── */

static int is_legal(GoBoardObject *b, int pos, int8_t color) {
    int size = b->size;
    if (b->board[pos] != 0) return 0;
    if (pos == b->ko_point)  return 0;

    int8_t opp = -color;
    int has_liberty    = 0;
    int captures_opp   = 0;
    int r = pos / size, c = pos % size;

    for (int d = 0; d < 4; d++) {
        int nr = r + DR[d], nc = c + DC[d];
        if (!in_bounds(size, nr, nc)) continue;
        int npos = idx(size, nr, nc);
        int8_t ncol = b->board[npos];

        if (ncol == 0) {
            has_liberty = 1;
        } else if (ncol == opp) {
            int root = dsu_find(b, npos);
            if (b->liberty_count[root] == 1) {
                captures_opp = 1;
            }
        } else if (ncol == color) {
            int root = dsu_find(b, npos);
            if (b->liberty_count[root] > 1) {
                has_liberty = 1;
            }
        }
    }
    return has_liberty || captures_opp;
}

/* ─── Remove group (capture) ─────────────────────────────────────────────── */

/*
 * Remove all stones of `root`'s group from the board.
 * Updates liberty counts of neighboring groups.
 * Stores captured positions in `out_captured` (caller must provide MAX_CELLS buf).
 * Returns number of captured stones.
 */
static int remove_group(GoBoardObject *b, int root, int *out_captured) {
    int size = b->size;
    int n_captured = 0;

    /* Collect stones to remove */
    int to_remove[MAX_CELLS];
    int nr_remove = 0;
    for (int pos = 0; pos < size * size; pos++) {
        if (b->board[pos] != 0 && dsu_find(b, pos) == root) {
            to_remove[nr_remove++] = pos;
        }
    }

    /* Remove stones and update neighbors */
    for (int i = 0; i < nr_remove; i++) {
        int pos = to_remove[i];
        int r = pos / size, c = pos % size;

        /* Update Zobrist */
        int col_idx = (b->board[pos] == 1) ? 0 : 1;
        b->zobrist_hash ^= b->zobrist_table[col_idx][pos];

        b->board[pos] = 0;
        /* Reset DSU for this cell */
        b->parent[pos] = pos;
        b->rank[pos]   = 0;
        b->liberty_count[pos] = 0;
        b->group_color[pos]   = 0;

        out_captured[n_captured++] = pos;
    }

    /* After removal, grant liberties to neighboring groups */
    for (int i = 0; i < nr_remove; i++) {
        int pos = to_remove[i];
        int r = pos / size, c = pos % size;

        for (int d = 0; d < 4; d++) {
            int nr2 = r + DR[d], nc2 = c + DC[d];
            if (!in_bounds(size, nr2, nc2)) continue;
            int npos = idx(size, nr2, nc2);
            if (b->board[npos] != 0) {
                /* This neighbor now has an extra liberty (the removed stone's position) */
                b->liberty_count[dsu_find(b, npos)]++;
            }
        }
    }

    return n_captured;
}

/* ─── play_move ───────────────────────────────────────────────────────────── */

static int play_move_internal(GoBoardObject *b, int pos, int8_t color) {
    int size = b->size;

    if (pos == -1) {
        /* Pass */
        b->consecutive_passes++;
        b->last_move = -1;
        b->ko_point  = -1;
        return 1;
    }

    if (!is_legal(b, pos, color)) return 0;

    /* Place stone */
    b->board[pos]  = color;
    b->parent[pos] = pos;
    b->rank[pos]   = 0;
    b->group_color[pos] = color;
    b->consecutive_passes = 0;
    b->last_move   = pos;

    /* Zobrist update */
    int col_idx = (color == 1) ? 0 : 1;
    b->zobrist_hash ^= b->zobrist_table[col_idx][pos];

    /* Count own immediate liberties (empty neighbors) */
    int r = pos / size, c = pos % size;
    int own_liberties = 0;
    for (int d = 0; d < 4; d++) {
        int nr = r + DR[d], nc = c + DC[d];
        if (!in_bounds(size, nr, nc)) continue;
        if (b->board[idx(size, nr, nc)] == 0) own_liberties++;
    }
    b->liberty_count[pos] = own_liberties;

    /* Merge with friendly neighbors, subtract shared liberty (pos) from them */
    int8_t opp = -color;
    for (int d = 0; d < 4; d++) {
        int nr = r + DR[d], nc = c + DC[d];
        if (!in_bounds(size, nr, nc)) continue;
        int npos = idx(size, nr, nc);
        if (b->board[npos] == color) {
            int nroot = dsu_find(b, npos);
            /* npos's group had `pos` as a liberty — remove it */
            b->liberty_count[nroot]--;
            dsu_union(b, pos, npos);
        }
    }

    /* Remove captured opponent groups */
    int total_captured = 0;
    int captured_buf[MAX_CELLS];
    int first_captured = -1;
    int n_first = 0;

    for (int d = 0; d < 4; d++) {
        int nr = r + DR[d], nc = c + DC[d];
        if (!in_bounds(size, nr, nc)) continue;
        int npos = idx(size, nr, nc);
        if (b->board[npos] == opp) {
            int oroot = dsu_find(b, npos);
            if (b->liberty_count[oroot] == 0) {
                int tmp[MAX_CELLS];
                int ncap = remove_group(b, oroot, tmp);
                if (first_captured == -1) {
                    first_captured = tmp[0];
                    n_first = ncap;
                    memcpy(captured_buf, tmp, ncap * sizeof(int));
                }
                total_captured += ncap;
            }
        }
    }

    /* Update capture counts */
    if (color == 1) b->black_captures += total_captured;
    else            b->white_captures += total_captured;

    /* Remove `pos` as a liberty from all remaining opponent neighbors
       (they were already updated in remove_group, but same-color neighbors
        that didn't get captured need to lose pos as liberty too) */
    /* Also: opponent neighbors that weren't captured lose `pos` as liberty */
    for (int d = 0; d < 4; d++) {
        int nr = r + DR[d], nc = c + DC[d];
        if (!in_bounds(size, nr, nc)) continue;
        int npos = idx(size, nr, nc);
        if (b->board[npos] == opp) {
            /* pos is now occupied — no longer a liberty */
            b->liberty_count[dsu_find(b, npos)]--;
        }
    }

    /* Ko detection: single stone captured and own group has one liberty */
    if (total_captured == 1 && n_first == 1) {
        int own_root = dsu_find(b, pos);
        if (b->liberty_count[own_root] == 1) {
            b->ko_point = captured_buf[0];
        } else {
            b->ko_point = -1;
        }
    } else {
        b->ko_point = -1;
    }

    return 1;
}

/* ─── Python type methods ─────────────────────────────────────────────────── */

static int GoBoard_init(GoBoardObject *self, PyObject *args, PyObject *kwds) {
    int size = 19;
    static char *kwlist[] = {"size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &size)) return -1;
    if (size < 2 || size > MAX_SIZE) {
        PyErr_SetString(PyExc_ValueError, "Board size must be 2-19");
        return -1;
    }
    self->size = size;
    init_zobrist(self);
    reset_board(self);
    return 0;
}

/* play_move(pos_or_none, color) */
static PyObject *GoBoard_play_move(GoBoardObject *self, PyObject *args) {
    PyObject *move_arg;
    int color;
    if (!PyArg_ParseTuple(args, "Oi", &move_arg, &color)) return NULL;

    int pos;
    if (move_arg == Py_None) {
        pos = -1;
    } else {
        int r, c;
        if (!PyArg_ParseTuple(move_arg, "ii", &r, &c)) return NULL;
        if (!in_bounds(self->size, r, c)) {
            PyErr_SetString(PyExc_ValueError, "Position out of bounds");
            return NULL;
        }
        pos = idx(self->size, r, c);
    }

    int ok = play_move_internal(self, pos, (int8_t)color);
    return PyBool_FromLong(ok);
}

/* is_legal_move(row, col, color) */
static PyObject *GoBoard_is_legal_move(GoBoardObject *self, PyObject *args) {
    int r, c, color;
    if (!PyArg_ParseTuple(args, "iii", &r, &c, &color)) return NULL;
    if (!in_bounds(self->size, r, c)) Py_RETURN_FALSE;
    int ok = is_legal(self, idx(self->size, r, c), (int8_t)color);
    return PyBool_FromLong(ok);
}

/* get_legal_moves(color) -> list of (r, c) */
static PyObject *GoBoard_get_legal_moves(GoBoardObject *self, PyObject *args) {
    int color;
    if (!PyArg_ParseTuple(args, "i", &color)) return NULL;

    PyObject *result = PyList_New(0);
    int size = self->size;
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            if (is_legal(self, idx(size, r, c), (int8_t)color)) {
                PyObject *move = Py_BuildValue("(ii)", r, c);
                PyList_Append(result, move);
                Py_DECREF(move);
            }
        }
    }
    return result;
}

/* to_numpy_array() -> bytes (raw int8, row-major, shape [size][size]) */
static PyObject *GoBoard_to_bytes(GoBoardObject *self, PyObject *args) {
    return PyBytes_FromStringAndSize(
        (const char *)self->board,
        self->size * self->size * sizeof(int8_t)
    );
}

/* copy() */
static PyObject *GoBoard_copy(GoBoardObject *self, PyObject *args) {
    GoBoardObject *new_board = PyObject_New(GoBoardObject, Py_TYPE(self));
    if (!new_board) return NULL;
    memcpy(new_board, self, sizeof(GoBoardObject));
    return (PyObject *)new_board;
}

/* Properties */
static PyObject *GoBoard_get_board_size(GoBoardObject *self, void *closure) {
    return PyLong_FromLong(self->size);
}
static PyObject *GoBoard_get_consecutive_passes(GoBoardObject *self, void *closure) {
    return PyLong_FromLong(self->consecutive_passes);
}
static PyObject *GoBoard_get_black_captures(GoBoardObject *self, void *closure) {
    return PyLong_FromLong(self->black_captures);
}
static PyObject *GoBoard_get_white_captures(GoBoardObject *self, void *closure) {
    return PyLong_FromLong(self->white_captures);
}
static PyObject *GoBoard_get_ko_point(GoBoardObject *self, void *closure) {
    if (self->ko_point < 0) Py_RETURN_NONE;
    int r = self->ko_point / self->size;
    int c = self->ko_point % self->size;
    return Py_BuildValue("(ii)", r, c);
}
static PyObject *GoBoard_get_last_move(GoBoardObject *self, void *closure) {
    if (self->last_move < 0) Py_RETURN_NONE;
    int r = self->last_move / self->size;
    int c = self->last_move % self->size;
    return Py_BuildValue("(ii)", r, c);
}
static PyObject *GoBoard_get_hash(GoBoardObject *self, void *closure) {
    return PyLong_FromUnsignedLongLong(self->zobrist_hash);
}
static PyObject *GoBoard_is_terminal(GoBoardObject *self, PyObject *args) {
    return PyBool_FromLong(self->consecutive_passes >= 2);
}

/* get_winner() - simplified scoring */
static PyObject *GoBoard_get_winner(GoBoardObject *self, PyObject *args) {
    if (self->consecutive_passes < 2) Py_RETURN_NONE;

    int size = self->size;
    int black_stones = 0, white_stones = 0;
    for (int i = 0; i < size * size; i++) {
        if (self->board[i] == 1)  black_stones++;
        if (self->board[i] == -1) white_stones++;
    }

    double black_score = black_stones + self->black_captures;
    double white_score = white_stones + self->white_captures + 6.5;

    if (black_score > white_score) return PyLong_FromLong(1);
    if (white_score > black_score) return PyLong_FromLong(-1);
    return PyLong_FromLong(0);
}

/* reset() */
static PyObject *GoBoard_reset(GoBoardObject *self, PyObject *args) {
    reset_board(self);
    Py_RETURN_NONE;
}

/* ─── Method/property tables ──────────────────────────────────────────────── */

static PyMethodDef GoBoard_methods[] = {
    {"play_move",       (PyCFunction)GoBoard_play_move,       METH_VARARGS, "Play a move"},
    {"is_legal_move",   (PyCFunction)GoBoard_is_legal_move,   METH_VARARGS, "Check legality"},
    {"get_legal_moves", (PyCFunction)GoBoard_get_legal_moves, METH_VARARGS, "Get legal moves"},
    {"to_bytes",        (PyCFunction)GoBoard_to_bytes,        METH_NOARGS,  "Raw board bytes"},
    {"copy",            (PyCFunction)GoBoard_copy,            METH_NOARGS,  "Deep copy"},
    {"is_terminal",     (PyCFunction)GoBoard_is_terminal,     METH_NOARGS,  "Is game over"},
    {"get_winner",      (PyCFunction)GoBoard_get_winner,      METH_NOARGS,  "Get winner"},
    {"reset",           (PyCFunction)GoBoard_reset,           METH_NOARGS,  "Reset board"},
    {NULL}
};

static PyGetSetDef GoBoard_getset[] = {
    {"size",              (getter)GoBoard_get_board_size,        NULL, "Board size", NULL},
    {"consecutive_passes",(getter)GoBoard_get_consecutive_passes,NULL, "Passes",     NULL},
    {"black_captures",    (getter)GoBoard_get_black_captures,    NULL, "B caps",     NULL},
    {"white_captures",    (getter)GoBoard_get_white_captures,    NULL, "W caps",     NULL},
    {"ko_point",          (getter)GoBoard_get_ko_point,          NULL, "Ko point",   NULL},
    {"last_move",         (getter)GoBoard_get_last_move,         NULL, "Last move",  NULL},
    {"zobrist_hash",      (getter)GoBoard_get_hash,              NULL, "Hash",       NULL},
    {NULL}
};

static PyTypeObject GoBoardType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "go_board.GoBoard",
    .tp_basicsize = sizeof(GoBoardObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "Fast Go board with Union-Find groups",
    .tp_new       = PyType_GenericNew,
    .tp_init      = (initproc)GoBoard_init,
    .tp_methods   = GoBoard_methods,
    .tp_getset    = GoBoard_getset,
};

/* ─── Module ──────────────────────────────────────────────────────────────── */

static PyModuleDef go_board_module = {
    PyModuleDef_HEAD_INIT,
    "go_board", "High-performance Go board (Union-Find)", -1, NULL
};

PyMODINIT_FUNC PyInit_go_board(void) {
    if (PyType_Ready(&GoBoardType) < 0) return NULL;
    PyObject *m = PyModule_Create(&go_board_module);
    if (!m) return NULL;
    Py_INCREF(&GoBoardType);
    PyModule_AddObject(m, "GoBoard", (PyObject *)&GoBoardType);
    return m;
}
