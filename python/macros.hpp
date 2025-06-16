#pragma once

// Defines a MAP macro function
// https://stackoverflow.com/questions/6707148/foreach-macro-on-macros-arguments
#define EVAL0(...) __VA_ARGS__
#define EVAL1(...) EVAL0(EVAL0(EVAL0(__VA_ARGS__)))
#define EVAL2(...) EVAL1(EVAL1(EVAL1(__VA_ARGS__)))
#define EVAL3(...) EVAL2(EVAL2(EVAL2(__VA_ARGS__)))
#define EVAL4(...) EVAL3(EVAL3(EVAL3(__VA_ARGS__)))
#define EVAL(...) EVAL4(EVAL4(EVAL4(__VA_ARGS__)))

#define MAP_OUT
#define MAP_END(...)
#define MAP_GET_END() 0, MAP_END
#define MAP_NEXT0(item, next, ...) next MAP_OUT
#define MAP_NEXT1(item, next) MAP_NEXT0(item, next, 0)
#define MAP_NEXT(item, next) MAP_NEXT1(MAP_GET_END item, next)

#define MAP_0(f, itr, x, peek, ...) f(itr, x) MAP_NEXT(peek, MAP_1)(f, itr + 1, peek, __VA_ARGS__)
#define MAP_1(f, itr, x, peek, ...) f(itr, x) MAP_NEXT(peek, MAP_0)(f, itr + 1, peek, __VA_ARGS__)
#define MAP(f, ...) EVAL(MAP_1(f, 0, __VA_ARGS__, (), 0))

#define MAP_C1_0(f, c1, itr, x, peek, ...) f(c1, itr, x) MAP_NEXT(peek, MAP_C1_1)(f, c1, itr + 1, peek, __VA_ARGS__)
#define MAP_C1_1(f, c1, itr, x, peek, ...) f(c1, itr, x) MAP_NEXT(peek, MAP_C1_0)(f, c1, itr + 1, peek, __VA_ARGS__)
#define MAP_C1(f, c1, ...) EVAL(MAP_C1_1(f, c1, 0, __VA_ARGS__, (), 0))

#define MAP_C2_0(f, c1, c2, itr, x, peek, ...) \
  f(c1, c2, itr, x) MAP_NEXT(peek, MAP_C2_1)(f, c1, c2, itr + 1, peek, __VA_ARGS__)
#define MAP_C2_1(f, c1, c2, itr, x, peek, ...) \
  f(c1, c2, itr, x) MAP_NEXT(peek, MAP_C2_0)(f, c1, c2, itr + 1, peek, __VA_ARGS__)
#define MAP_C2(f, c1, c2, ...) EVAL(MAP_C2_1(f, c1, c2, 0, __VA_ARGS__, (), 0))

#define FOLD_OUT
#define FOLD_END(f, val, ...) val
#define FOLD_GET_END() 0, FOLD_END
#define FOLD_NEXT0(item, next, ...) next FOLD_OUT
#define FOLD_NEXT1(item, next) FOLD_NEXT0(item, next, 0)
#define FOLD_NEXT(item, next) FOLD_NEXT1(FOLD_GET_END item, next)

#define FOLD_0(f, res, x, peek, ...) FOLD_NEXT(peek, FOLD_1)(f, f(res, x), peek, __VA_ARGS__)
#define FOLD_1(f, res, x, peek, ...) FOLD_NEXT(peek, FOLD_0)(f, f(res, x), peek, __VA_ARGS__)
#define FOLD(f, init, ...) EVAL(FOLD_1(f, init, __VA_ARGS__, (), 0))

#define COUNT_INNER(i, unused) i + 1
#define COUNT(...) FOLD(COUNT_INNER, 0, __VA_ARGS__)

#define PACK_TUPLE_INNER(obj, unused, name) , obj.name
#define PACK_TUPLE_1(obj, name0, ...) py::make_tuple(obj.name0 MAP_C1(PACK_TUPLE_INNER, obj, __VA_ARGS__))
#define PACK_TUPLE(obj, ...) PACK_TUPLE_1(obj, __VA_ARGS__)

#define UNPACK_TUPLE_INNER(obj_type, tuple, itr, name) , .name = tuple[itr + 1].cast<decltype(obj_type::name)>()
#define UNPACK_TUPLE_1(obj_type, tuple, name0, ...)                                                              \
  obj_type {                                                                                                     \
    .name0 = tuple[0].cast<decltype(obj_type::name0)>() MAP_C2(UNPACK_TUPLE_INNER, obj_type, tuple, __VA_ARGS__) \
  }
#define UNPACK_TUPLE(obj_type, tuple, ...) UNPACK_TUPLE_1(obj_type, tuple, __VA_ARGS__)
