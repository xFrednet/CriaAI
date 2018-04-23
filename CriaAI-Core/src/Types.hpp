#pragma once

#include <string>
typedef std::string                    String;

#include <memory>
template<typename T>
using cr_ptr                           = std::shared_ptr<T>;


typedef char                           int8;
typedef short                          int16;
typedef int                            int32;
typedef long long                      int64;

typedef unsigned char                  uint8;
typedef unsigned short                 uint16;
typedef unsigned int                   uint32;
typedef unsigned long long             uint64;

typedef uint32                         uint;
typedef unsigned char                  byte;
