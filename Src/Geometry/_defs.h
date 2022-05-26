#ifndef ___DEFS_H
#define ___DEFS_H

/*  _DEFS.H

    Most common data types, calling conventions,
    memory model dependent macros and other stuff.

    Copyright (c) 1993 by Octopus Ltd.
    All rights reserved.
*/

#pragma pack (push, 4)

typedef double REAL;

typedef signed char     schar;
typedef signed short    sshort;
typedef signed int      sint;
typedef signed long     slong;

typedef unsigned char   uchar;
typedef unsigned short  ushort;
typedef unsigned int    uint;
typedef unsigned long   ulong;

typedef uchar   BYTE;
typedef ushort  WORD;
typedef ulong   DWORD;
typedef uchar   byte;
typedef ushort  word;
typedef ulong   dword;

typedef ushort  PHINDEX;
#define MAXPHINDEX      0xFFFF

typedef uchar   COLOR;

typedef int     logical;

typedef void * POINTER;


#define NO 0
#define YES 1

#define DECLARE extern
#define GLOBAL  extern
#define PRIVATE static
#define PUBLIC
#define EXTERN  extern

#ifdef __PROJECT_ONLY
#define _Cdecl  cdecl
#define _Pascal pascal
#else
#define _Cdecl
#define _Pascal
#endif

#define _FARCALL far
#define _FARFUNC
#define _FAR

# define DTSIZE(x)

#define _CLASSTYPE  near

#ifdef  __cplusplus
#define _PTRDEF(name)           typedef name _FAR * P##name;
#define _REFDEF(name)           typedef name _FAR & R##name;
#define _REFPTRDEF(name)        typedef name _FAR * _FAR & RP##name;
#define _PTRCONSTDEF(name)      typedef const name _FAR * PC##name;
#define _REFCONSTDEF(name)      typedef const name _FAR & RC##name;
#define _CLASSDEF(name)         class   _CLASSTYPE name;        \
                                        _PTRDEF(name)           \
                                        _REFDEF(name)           \
                                        _REFPTRDEF(name)        \
                                        _PTRCONSTDEF(name)      \
                                        _REFCONSTDEF(name)

class OBJECT;
typedef void ( * OBJFUNC ) ( OBJECT * );

#endif  /*  __cplusplus  */

#ifdef  __cplusplus
extern "C" void  MAIN ( void );
#else
extern  void  MAIN ( void );
#endif

// ����� � ����

#define SETBIT(dst,bit,v)       dst = (dst & ~(1<<bit)) | ( (v) ? (1<<bit) : 0 )
#define GETBIT(dst,bit)         (dst & (1<<bit)) ? YES : NO

#pragma pack (pop)

#endif  /*  ___DEFS_H   */
