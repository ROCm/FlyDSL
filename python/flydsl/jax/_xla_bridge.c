/*
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2025 FlyDSL Project Contributors
 *
 * Thin C trampoline that bridges XLA's GPU custom-call convention to
 * FlyDSL's bare-pointer convention.
 *
 * XLA GPU custom call (API_VERSION_STATUS_RETURNING_UNIFIED):
 *   void fn(hipStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
 *
 * FlyDSL bare-pointer convention:
 *   void fn(void** ptrs)
 *   where ptrs[i] = &storage[i], storage[i] = device_ptr or stream value
 *
 * Compiled at import time via: cc -shared -fPIC -o _xla_bridge.so _xla_bridge.c -lpthread
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>

#define MAX_BUFFERS 64
#define MAX_SCALARS 16
#define MAX_TARGETS 256

typedef void (*flydsl_func_t)(void **ptrs);

typedef struct {
    flydsl_func_t func;
    int n_buffers;
    int n_scalars;
    int64_t scalar_vals[MAX_SCALARS];
} target_slot_t;

static target_slot_t g_targets[MAX_TARGETS];
static int g_n_targets = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

int flydsl_xla_register(void *func_ptr, int n_buffers, int n_scalars,
                        int64_t *scalar_vals) {
    if (n_buffers > MAX_BUFFERS || n_scalars > MAX_SCALARS)
        return -1;

    pthread_mutex_lock(&g_lock);
    if (g_n_targets >= MAX_TARGETS) {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    int idx = g_n_targets++;
    /* Populate slot while holding lock so concurrent dispatchers never
       observe an allocated index with uninitialized data. */
    g_targets[idx].func = (flydsl_func_t)func_ptr;
    g_targets[idx].n_buffers = n_buffers;
    g_targets[idx].n_scalars = n_scalars;
    for (int i = 0; i < n_scalars; i++)
        g_targets[idx].scalar_vals[i] = scalar_vals ? scalar_vals[i] : 0;
    pthread_mutex_unlock(&g_lock);

    return idx;
}

static void xla_bridge_dispatch(void *stream, void **buffers,
                                const char *opaque, size_t opaque_len) {
    /* Validate opaque: must be exactly sizeof(int) carrying the slot index. */
    if (opaque == NULL || opaque_len != sizeof(int))
        return;

    int idx;
    memcpy(&idx, opaque, sizeof(int));

    if (idx < 0 || idx >= g_n_targets)
        return;

    target_slot_t *t = &g_targets[idx];
    int nb = t->n_buffers;
    int ns = t->n_scalars;

    /* Build FlyDSL's ptrs array on the stack.
     * Layout: [buf0, buf1, ..., scalar0, scalar1, ..., stream]
     *
     * For scalars, FlyDSL reads through ptrs[i] -> &storage[i] as a
     * pointer-sized value.  We store the int64_t in a dedicated array
     * and point into it, avoiding integer-to-pointer casts.
     */
    void *storage[MAX_BUFFERS + 1];
    int64_t scalar_storage[MAX_SCALARS];
    void *packed[MAX_BUFFERS + MAX_SCALARS + 1];

    /* Tensor buffers. */
    for (int i = 0; i < nb; i++) {
        storage[i] = buffers[i];
        packed[i] = &storage[i];
    }
    /* Scalar values (from the registered slot, not from opaque). */
    for (int i = 0; i < ns; i++) {
        scalar_storage[i] = t->scalar_vals[i];
        packed[nb + i] = &scalar_storage[i];
    }
    /* Stream in the last slot. */
    storage[nb] = stream;
    packed[nb + ns] = &storage[nb];

    t->func(packed);
}

void *flydsl_xla_get_bridge(int idx) {
    (void)idx;
    return (void *)&xla_bridge_dispatch;
}
