#pragma once

#ifndef COMFAIX_EXPORTS
#if (defined _WIN32 || defined WINCE || defined __CYGWIN__) &&                 \
    defined(COMFAIXAPI_EXPORTS)
#define COMFAIX_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4 &&                                     \
    (defined(COMFAIXAPI_EXPORTS) || defined(__APPLE__))
#define COMFAIX_EXPORTS __attribute__((visibility("default")))
#endif
#endif

#ifndef COMFAIX_EXPORTS
#define COMFAIX_EXPORTS
#endif
