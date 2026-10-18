#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*api_callback)(void *, unsigned, unsigned, const void *);
typedef void (*request_callback)(uint8_t **, size_t *, size_t *);
typedef void (*complete_callback)(void *, unsigned, uint8_t *, size_t, size_t);

static api_callback callback;
static request_callback requested;
static complete_callback completed;
static unsigned enabled, inside_exit, finalized;
static void *current_context = (void *)(uintptr_t)7;

static void record(const char *name) {
    FILE *log = fopen(getenv("D7_FAKE_LOG"), "a");
    if (!log) abort();
    fprintf(log, "%s\n", name);
    fclose(log);
}

int cuptiComputeCapabilitySupported(int major, int minor, int *supported) {
    *supported = major == 6 && minor == 1;
    return 0;
}

int cuptiSubscribe(void **subscriber, api_callback cb, void *userdata) {
    (void)userdata;
    callback = cb;
    *subscriber = (void *)(uintptr_t)1;
    record("subscribe");
    return 0;
}

int cuptiUnsubscribe(void *subscriber) {
    (void)subscriber;
    record("unsubscribe");
    return 0;
}

int cuptiActivityRegisterCallbacks(request_callback req, complete_callback done) {
    requested = req;
    completed = done;
    return 0;
}

int cuptiActivityEnable(unsigned kind) {
    (void)kind;
    enabled++;
    return 0;
}

int cuptiActivityDisable(unsigned kind) {
    (void)kind;
    if (!enabled || finalized) abort();
    enabled--;
    record("disable");
    return 0;
}

int cuptiActivityFlushAll(unsigned flags) {
    (void)flags;
    if (finalized || enabled) abort();
    record("flush");
    uint8_t *buffer = NULL;
    size_t size = 0, max_records = 0;
    requested(&buffer, &size, &max_records);
    if (!buffer || !size || max_records) abort();
    completed(NULL, 0, buffer, size, 0);
    return getenv("D7_FAKE_FLUSH_ERROR") ? 1 : 0;
}

int cuptiActivityGetNumDroppedRecords(void *context, unsigned stream, size_t *dropped) {
    (void)context;
    (void)stream;
    *dropped = 0;
    return 0;
}

int cuptiActivityGetNextRecord(uint8_t *buffer, size_t size, void **record_out) {
    (void)buffer;
    (void)size;
    (void)record_out;
    abort();
}

int cuptiGetResultString(int result, const char **message) {
    (void)result;
    *message = "synthetic CUPTI error";
    return 0;
}

int cuptiEnableCallback(unsigned enable, void *subscriber, unsigned domain, unsigned cbid) {
    (void)subscriber;
    if (!enable || domain != 1 || cbid != 1 || finalized || !callback) abort();
    record("enable_exit");
    return 0;
}

int cuInit(unsigned flags) {
    (void)flags;
    record("cuInit");
    if (!getenv("D7_FAKE_SKIP_CALLBACK")) {
        struct { unsigned callback_site; void *rest[16]; } data = {0};
        callback(NULL, 1, 1, &data);
        data.callback_site = 1;
        inside_exit = 1;
        callback(NULL, 1, 1, &data);
        inside_exit = 0;
    }
    return 0;
}

int cuDeviceGetCount(int *count) { *count = 2; return 0; }
int cuDeviceGet(int *device, int ordinal) { *device = ordinal; return 0; }
int cuDevicePrimaryCtxRetain(void **context, int device) {
    *context = (void *)(uintptr_t)(100 + device);
    record("retain");
    return 0;
}
int cuCtxGetCurrent(void **context) { *context = current_context; return 0; }
int cuCtxSetCurrent(void *context) {
    current_context = context;
    record(context == (void *)(uintptr_t)7 ? "restore" : "bind");
    return 0;
}
int cuCtxSynchronize(void) {
    if (enabled || (uintptr_t)current_context < 100) abort();
    record("synchronize");
    return getenv("D7_FAKE_SYNC_ERROR") ? 1 : 0;
}
int cuDevicePrimaryCtxRelease_v2(int device) {
    (void)device;
    if (current_context != (void *)(uintptr_t)7) abort();
    record("release");
    return 0;
}
int cuGetErrorString(int result, const char **message) {
    (void)result;
    *message = "synthetic CUDA error";
    return 0;
}

int cuptiFinalize(void) {
    if (!inside_exit || enabled || finalized) abort();
    record("finalize_at_exit");
    if (getenv("D7_FAKE_FINALIZE_ERROR")) return 1;
    finalized = 1;
    callback = NULL;
    return 0;
}
