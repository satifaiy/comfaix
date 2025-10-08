# ******************* json ********************

FetchContent_Declare(
    simdjson
    GIT_REPOSITORY https://github.com/simdjson/simdjson.git
    GIT_TAG        v4.0.7
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(simdjson)
