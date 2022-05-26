/*  SIMILARITY.H

    Vector Library Include File
    The 3-dim similarity transformations class.

    Copyright (c) 1993 by Octopus Ltd.
    All Rights Reserved.
*/

#ifndef __SIMILARITY_H
#define __SIMILARITY_H

#ifndef __cplusplus
#error C++ compiler required
#endif

#include "_defs.h"
#include "Real.h"
#include "Vector.h"
#include "Matrix.h"
#include "Point.h"
#include "Conform.h"

#pragma pack (push, 4)

struct  SIMILARITY
{
  // -------- Implementation -------- //
        MATRIX  rot;
        VECTOR3 trans;
  // -------------------------------- //

  // -- Constructors
        SIMILARITY ( );
        SIMILARITY ( const CONFORM & );

  // -- Operators

        POINT3  operator () ( const POINT3 & ) const;

};

#pragma pack (pop)

#endif  //  __SIMILARITY_H
