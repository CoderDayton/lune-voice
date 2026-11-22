#include <include/whisper.h>
/* LUNE FORK: the Parakeet TDT runtime whisper.cpp ships beside whisper. */
#include <include/parakeet.h>
/* LUNE FORK: ggml comes from the vendored llama.cpp tree. */
#include <ggml.h>

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
