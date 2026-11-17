// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// Windows stand-in for the part of <pthread.h> that FlyRocmAotRuntime.cpp uses,
// built on kernel32's slim reader/writer locks. A zeroed SRWLOCK is unlocked
// and is never destroyed.

#ifndef FLYDSL_RUNTIME_ROCM_WINDOWS_PTHREAD_H
#define FLYDSL_RUNTIME_ROCM_WINDOWS_PTHREAD_H

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

typedef SRWLOCK pthread_mutex_t;
typedef void pthread_mutexattr_t;

// An SRW lock is released by the call that matches how it was taken, so the
// rwlock records its exclusive owner.
typedef struct {
  SRWLOCK lock;
  DWORD writer;
} pthread_rwlock_t;

// clang-format 18.1.3 and 18.1.8 disagree on this line; keep it as is.
// clang-format off
#define PTHREAD_RWLOCK_INITIALIZER {SRWLOCK_INIT, 0}
// clang-format on

static inline int pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *) {
  InitializeSRWLock(mutex);
  return 0;
}

static inline int pthread_mutex_destroy(pthread_mutex_t *) { return 0; }

static inline int pthread_mutex_lock(pthread_mutex_t *mutex) {
  AcquireSRWLockExclusive(mutex);
  return 0;
}

static inline int pthread_mutex_unlock(pthread_mutex_t *mutex) {
  ReleaseSRWLockExclusive(mutex);
  return 0;
}

static inline int pthread_rwlock_rdlock(pthread_rwlock_t *rwlock) {
  AcquireSRWLockShared(&rwlock->lock);
  return 0;
}

static inline int pthread_rwlock_wrlock(pthread_rwlock_t *rwlock) {
  AcquireSRWLockExclusive(&rwlock->lock);
  rwlock->writer = GetCurrentThreadId();
  return 0;
}

// Only the exclusive owner finds its own id here: thread ids are never 0, and
// while any thread holds the lock shared, no thread holds it exclusively.
static inline int pthread_rwlock_unlock(pthread_rwlock_t *rwlock) {
  if (rwlock->writer == GetCurrentThreadId()) {
    rwlock->writer = 0;
    ReleaseSRWLockExclusive(&rwlock->lock);
  } else {
    ReleaseSRWLockShared(&rwlock->lock);
  }
  return 0;
}

#endif // FLYDSL_RUNTIME_ROCM_WINDOWS_PTHREAD_H
