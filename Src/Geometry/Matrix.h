#ifndef __MATRIX_H
#define __MATRIX_H

/*  MATRIX.H

    Vector Library Include File
    Declarations for 3-dim MATRIXes.

    Copyright (c) 1993 by Octopus Ltd.
    All Rights Reserved.
*/

#ifndef __cplusplus
#error C++ compiler required
#endif

#include "_defs.h"
#include "Real.h"
#include "Vector.h"
#include "Spin.h"

#pragma pack (push, 4)

struct  SPIN;

struct  MATRIX
{
  // -------- Implementation -------- //
        VECTOR3 x, y, z;
  // -------------------------------- //

  // -- Constructors
        MATRIX  ( );
        MATRIX  ( const MATRIX & );
        MATRIX  ( const SPIN & );
        MATRIX  ( const VECTOR3 & , const VECTOR3 & , const VECTOR3 & );
        MATRIX  (
                    REAL , REAL , REAL ,
                    REAL , REAL , REAL ,
                    REAL , REAL , REAL
                );

  // -- Indexed access operator
        VECTOR3 &       operator [] ( int );
        const VECTOR3 & operator [] ( int ) const;

  // -- Addition & Subtraction operators
        MATRIX      operator +  ( ) const;
        MATRIX      operator -  ( ) const;
        MATRIX &    operator += ( const MATRIX & );
        MATRIX &    operator -= ( const MATRIX & );

  // -- Multiplication by REAL number operators
        MATRIX &    operator *= ( REAL );
        MATRIX &    operator /= ( REAL );

  // -- Miscellaneous operators & functions
        MATRIX      operator ~  ( ) const;
        MATRIX      operator !  ( ) const;
        REAL        I       ( ) const;
        REAL        II      ( ) const;
        REAL        III     ( ) const;
        REAL        tr      ( ) const;
        REAL        det     ( ) const;
        MATRIX      conj    ( ) const;
        MATRIX      comp    ( ) const;
        MATRIX      inv     ( ) const;
        VECTOR3     col     ( int ) const;
        VECTOR3     row     ( int ) const;

};


// ------------------------------------ //
//  Out-of-class operators & functions  //
// ------------------------------------ //

logical operator == ( const MATRIX  & , const MATRIX & );
logical operator != ( const MATRIX  & , const MATRIX & );
MATRIX  operator +  ( const MATRIX  & , const MATRIX & );
MATRIX  operator -  ( const MATRIX  & , const MATRIX & );
MATRIX  operator *  ( const MATRIX  & , REAL );
MATRIX  operator /  ( const MATRIX  & , REAL );
MATRIX  operator *  ( REAL , const MATRIX & );
VECTOR3 operator |  ( const MATRIX  & , const VECTOR3 & );
VECTOR3 operator |  ( const VECTOR3 & , const MATRIX  & );
MATRIX  operator |  ( const MATRIX  & , const MATRIX  & );
MATRIX  operator &  ( const MATRIX  & , const VECTOR3 & );
MATRIX  operator &  ( const VECTOR3 & , const MATRIX  & );
MATRIX  operator ^  ( const VECTOR3 & , const VECTOR3 & );

REAL    tr      ( const MATRIX & );
REAL    det     ( const MATRIX & );


DECLARE const MATRIX MATRIX_0;
DECLARE const MATRIX MATRIX_1;


// ------------------------------------------------ //
//  Inline definitions for some 'dummy' functions   //
// ------------------------------------------------ //


inline  VECTOR3 &  MATRIX:: operator [] ( int index )
{
    return  ( ( VECTOR3 * ) ( void * ) this ) [ index ];
}


inline  const VECTOR3 &  MATRIX:: operator [] ( int index ) const
{
    return  ( ( const VECTOR3 * ) ( const void * ) this ) [ index ];
}

#pragma pack (pop)

#endif  //  __MATRIX_H
