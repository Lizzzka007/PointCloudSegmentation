/*  VectorLib.cpp

    Copyright (c) 1996 by Octopus Ltd.
    All rights reserved.
*/

// #include "StdAfx.h"
#include "math.h"
//#include "../include/tune.h"
//#include "../include/vector2.h"
//#include "../include/similart.h"
#include "Tune.h"
#include "Vector2.h"
#include "Similart.h"

const int RM_NOTHING = 0x00;
const int RM_ROTATE  = 0x01;
const int RM_MOVE    = 0x02;
const int RM_OBJECT  = 0x04;
const int RM_SIDE    = 0x08;
const int RM_IMAGE   = 0x10;
const int RM_ALL     = 0x1F;

extern  const VECTOR2 VECTOR2_0 (0,0);

VECTOR2:: VECTOR2 () { }

VECTOR2:: VECTOR2 (const VECTOR2 &v) { *this=v; }

VECTOR2:: VECTOR2 (REAL X, REAL Y) { x=X; y=Y; }

VECTOR2  VECTOR2:: operator+ () const { return *this; }

VECTOR2  VECTOR2:: operator- () const { return VECTOR2 (-x,-y); }

VECTOR2 &  VECTOR2:: operator+= (const VECTOR2 &v)
{ x+=v.x; y+=v.y;  return *this; }

VECTOR2 &  VECTOR2:: operator-= (const VECTOR2 &v)
{ x-=v.x; y-=v.y;  return *this; }

VECTOR2 &  VECTOR2:: operator*= (REAL r)
{ x*=r; y*=r;  return *this; }

VECTOR2 &  VECTOR2:: operator/= (REAL r)
{ x/=r; y/=r;  return *this; }

REAL  VECTOR2:: abs () const { return sqrt (x*x+y*y); }

VECTOR2  VECTOR2:: fix (REAL r) const
{ return (*this)*(r/abs()); }

VECTOR2  VECTOR2:: project (const VECTOR2 &v) const
{ return ((v|*this)/(*this|*this)) * (*this); }


logical  operator== (const VECTOR2 &v1, const VECTOR2 &v2)
{ return v1.x==v2.x && v1.y==v2.y; }

logical  operator!= (const VECTOR2 &v1, const VECTOR2 &v2)
{ return v1.x!=v2.x || v1.y!=v2.y; }

VECTOR2  operator+ (const VECTOR2 &v1, const VECTOR2 &v2)
{ return VECTOR2 (v1.x+v2.x, v1.y+v2.y); }

VECTOR2  operator- (const VECTOR2 &v1, const VECTOR2 &v2)
{ return VECTOR2 (v1.x-v2.x, v1.y-v2.y); }

VECTOR2  operator* (const VECTOR2 &v, REAL r)
{ return VECTOR2 (v.x*r, v.y*r); }

VECTOR2  operator/ (const VECTOR2 &v, REAL r)
{ return VECTOR2 (v.x/r, v.y/r); }

VECTOR2  operator* (REAL r, const VECTOR2 &v)
{ return VECTOR2 (v.x*r, v.y*r); }

REAL  operator| (const VECTOR2 &v1, const VECTOR2 &v2)
{ return v1.x*v2.x + v1.y*v2.y; }

REAL  operator& (const VECTOR2 &v1, const VECTOR2 &v2)
{ return v1.x*v2.y - v1.y*v2.x; }

REAL  abs (const VECTOR2 &v)
{ return sqrt (v.x*v.x+v.y*v.y); }

extern  const VECTOR VECTOR_0 (0,0,0);

VECTOR:: VECTOR () { }

VECTOR:: VECTOR (const VECTOR &v) { *this=v; }

VECTOR:: VECTOR (REAL X, REAL Y, REAL Z) { x=X; y=Y; z=Z; }

VECTOR  VECTOR:: operator+ () const { return *this; }

VECTOR  VECTOR:: operator- () const { return VECTOR (-x,-y,-z); }

VECTOR &  VECTOR:: operator+= (const VECTOR &v)
{ x+=v.x; y+=v.y; z+=v.z; return *this; }

VECTOR &  VECTOR:: operator-= (const VECTOR &v)
{ x-=v.x; y-=v.y; z-=v.z; return *this; }

VECTOR &  VECTOR:: operator*= (REAL r)
{ x*=r; y*=r; z*=r; return *this; }

VECTOR &  VECTOR:: operator/= (REAL r)
{ x/=r; y/=r; z /=r; return *this; }

REAL  VECTOR:: abs () const { return sqrt (x*x+y*y+z*z); }

VECTOR  VECTOR:: fix (REAL r) const
{ return (*this)*(r/abs()); }

VECTOR  VECTOR:: project (const VECTOR &v) const
{ return ((v|*this)/(*this|*this)) * (*this); }

logical  operator== (const VECTOR &v1, const VECTOR &v2)
{ return v1.x==v2.x && v1.y==v2.y && v1.z==v2.z; }

logical  operator!= (const VECTOR &v1, const VECTOR &v2)
{ return v1.x!=v2.x || v1.y!=v2.y || v1.z !=v2.z; }

VECTOR  operator+ (const VECTOR &v1, const VECTOR &v2)
{ return VECTOR (v1.x+v2.x, v1.y+v2.y, v1.z+v2.z); }

VECTOR  operator- (const VECTOR &v1, const VECTOR &v2)
{ return VECTOR (v1.x-v2.x, v1.y-v2.y, v1.z-v2.z); }

VECTOR  operator* (const VECTOR &v, REAL r)
{ return VECTOR (v.x*r, v.y*r, v.z*r); }

VECTOR  operator/ (const VECTOR &v, REAL r)
{ return VECTOR (v.x/r, v.y/r, v.z/r); }

VECTOR  operator* (REAL r, const VECTOR &v)
{ return VECTOR (v.x*r, v.y*r, v.z*r); }

REAL  operator| (const VECTOR &v1, const VECTOR &v2)
{ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }

VECTOR  operator& (const VECTOR &v1, const VECTOR &v2)
{
    VECTOR res;
    res.x = v1.y*v2.z-v1.z*v2.y;
    res.y = v1.z*v2.x-v1.x*v2.z;
    res.z = v1.x*v2.y-v1.y*v2.x;
    return res;
}

REAL  abs (const VECTOR &v)
{ return sqrt (v.x*v.x+v.y*v.y+v.z*v.z); }


#define EPS     1e-64

static void norm ( SPIN & sp )
{
    REAL s = sqrt ( sp.t * sp.t + sp.x * sp.x + sp.y * sp.y + sp.z * sp.z );
    sp.t /= s;
    sp.x /= s;
    sp.y /= s;
    sp.z /= s;
    if ( fabs ( sp.t ) < EPS ) sp.t = 0;
    if ( fabs ( sp.x ) < EPS ) sp.x = 0;
    if ( fabs ( sp.y ) < EPS ) sp.y = 0;
    if ( fabs ( sp.z ) < EPS ) sp.z = 0;
}

void getspin ( SPIN & s, const VECTOR & a, const VECTOR & b, const VECTOR * n )
{
    VECTOR norm ( a & b );
    double scal = a | b;
    double angle;
    if ( fabs ( scal ) > 0.999999999999 )
    {
        if ( n ) norm = * n;
        else
        {
            VECTOR tmp;
            //supplement_reper ( a, norm, tmp );
        }
        angle = ( scal > 0 ) ? 0 : M_PI;
    }
    else angle = acos ( scal );
    s = SPIN ( norm, angle );
}

CONFORM:: CONFORM ()
{}

CONFORM:: CONFORM ( const SPIN & s, const VECTOR & t, REAL m ) :
spin    (s),
trans   (t),
magn    (m)
{}

CONFORM CONFORM:: operator ~ ( ) const
{
    SPIN s ( ~spin );
    REAL m = 1/magn;
    return CONFORM ( s, (-m) * s ( trans ), m );
}

PLANE CONFORM:: operator () ( const PLANE & p ) const
{
    VECTOR norm ( spin ( p.norm ) );
    return PLANE ( norm, magn * p.dist - ( norm | trans ) );
}

CONFORM operator >> ( const CONFORM & b1, const CONFORM & b2 )
{
    SPIN sp ( b1.spin * b2.spin );
    norm ( sp );
    return CONFORM
    (
        sp,
        b1.spin ( b2.trans ) * b1.magn + b1.trans,
        b1.magn * b2.magn
    );
}

BASIS operator >> ( const BASIS & b1, const BASIS & b2 )
{
    SPIN sp ( b1.spin * b2.spin );
    norm ( sp );
    return BASIS
    (
        sp,
        b1.spin ( b2.trans ) * b1.magn + b1.trans,
        b1.magn * b2.magn,
        RM_ROTATE | b1.alt | b2.alt
    );
}

BASIS operator >> ( const SPIN & s, const BASIS & b )
{
    return BASIS ( s * b.spin, b.trans, b.magn, RM_ROTATE | b.alt );
}

BASIS operator >> ( const VECTOR & v, const BASIS & b )
{
    return BASIS ( b.spin, v + b.trans, b.magn, RM_MOVE | b.alt );
}

BASIS operator >> ( const REAL & m, const BASIS & b )
{
    return BASIS ( b.spin, b.trans, m * b.magn, RM_OBJECT | b.alt );
}

BASIS:: BASIS ( const CONFORM & c, int a ) :
CONFORM ( c ),
alt ( a )
{
}

BASIS:: BASIS ( const SPIN & s, const VECTOR & v, REAL r , int a ) :
CONFORM ( s, v, r ),
alt ( a )
{ }

BASIS:: BASIS ( ) : CONFORM ( SPIN_1, VECTOR_0, 1 ), alt ( 0 ) { }

BASIS operator + ( const BASIS & b1, const BASIS & b2 )
{
    return BASIS ( b1.spin * b2.spin, b1.trans + b2.trans, 1 );
}

static SPIN prepare_spin ( const SPIN & s, REAL a )
{
    REAL t2 = s.t * s.t;
    if ( t2 > 0.99999999 ) return s;
    t2 = sqrt ( 1 - t2 );
    REAL angle = 2 * a * atan2 ( s.t, t2 );
    return SPIN ( VECTOR (s.x, s.y, s.z), angle );
}

BASIS operator * ( REAL a, const BASIS & b )
{
    return BASIS ( prepare_spin ( b.spin, a ), b.trans * a, 1 );
}

DECLARE const CONFORM CONFORM_1 ( SPIN ( VECTOR ( 1,0,0 ), 0 ), VECTOR ( 0,0,0 ), 1 );
DECLARE const BASIS BASIS_1   ( CONFORM_1 );

PLANE:: PLANE   ( ) { }

PLANE:: PLANE   ( const VECTOR & v, REAL d )
{
    norm = v;   dist = d;
}

ushort PLANE:: light ( const VECTOR & v ) const
{
    REAL vol = norm | v;
    return ( vol > 0 ) ? 2 : ( ( vol < 0 ) ? 0 : 1 );
}

PLANE PLANE:: operator -  ( ) const
{
    return PLANE ( -norm, - dist );
}

SIMILARITY:: SIMILARITY ( ) { }

SIMILARITY:: SIMILARITY ( const CONFORM & c )
{
    rot = c.magn * MATRIX ( c.spin );
    trans = c.trans;
}

POINT3   SIMILARITY:: operator () ( const POINT3 & p ) const
{
    return ( rot | p ) + trans;
}

SPIN::SPIN ( const VECTOR & axis, REAL angle )
{
    REAL a, ac, as;
    VECTOR f;
    a = angle / 2;
    ac = cos (a);
    as = sin (a);
    f = axis.fix ();
    t = ac;
    x = as * f.x;
    y = as * f.y;
    z = as * f.z;
}

MATRIX::MATRIX ( const SPIN & s )
{
    REAL tt,uu,vv,ww,vw,wu,uv,ut,vt,wt;

    tt=s.t*s.t; uu=s.x*s.x; vv=s.y*s.y; ww=s.z*s.z;
    vw=s.y*s.z; wu=s.z*s.x; uv=s.x*s.y; ut=s.x*s.t; vt=s.y*s.t; wt=s.z*s.t;
    x.x=tt+uu-vv-ww; x.y=2*(uv-wt); x.z=2*(wu+vt);
    y.y=tt+vv-ww-uu; y.z=2*(vw-ut); y.x=2*(uv+wt);
    z.z=tt+ww-uu-vv; z.x=2*(wu-vt); z.y=2*(vw+ut);
}

SPIN:: SPIN () { }

SPIN:: SPIN ( const SPIN & s)
{
    t = s.t; x = s.x; y = s.y; z = s.z;
}


SPIN SPIN::operator~() const
{
    SPIN s;
    s.t = t; s.x = -x; s.y = -y; s.z = -z;
    return s;
}

SPIN operator * ( const SPIN & s1, const SPIN & s2 )
{
    SPIN s;
    s.t = s1.t*s2.t - s1.x*s2.x - s1.y*s2.y - s1.z*s2.z;
    s.x = s1.t*s2.x + s1.x*s2.t + s1.y*s2.z - s1.z*s2.y;
    s.y = s1.t*s2.y + s1.y*s2.t + s1.z*s2.x - s1.x*s2.z;
    s.z = s1.t*s2.z + s1.z*s2.t + s1.x*s2.y - s1.y*s2.x;
    return s;
}

VECTOR SPIN::operator()(const VECTOR &o) const
{
    VECTOR res;
    REAL tt,uu,vv,ww,vw,wu,uv,ut,vt,wt;

    tt=t*t; uu=x*x; vv=y*y; ww=z*z;
    vw=y*z; wu=z*x; uv=x*y; ut=x*t; vt=y*t; wt=z*t;
    res.x = (tt+uu-vv-ww)*o.x + 2*((uv-wt)*o.y+(wu+vt)*o.z);
    res.y = (tt+vv-ww-uu)*o.y + 2*((vw-ut)*o.z+(uv+wt)*o.x);
    res.z = (tt+ww-uu-vv)*o.z + 2*((wu-vt)*o.x+(vw+ut)*o.y);
    return res;
}

MATRIX SPIN::operator()(const MATRIX &o) const
{
    MATRIX res;
    REAL tt,uu,vv,ww,vw,wu,uv,ut,vt,wt;
    REAL rxx,rxy,rxz,ryx,ryy,ryz,rzx,rzy,rzz;
    REAL orxx,orxy,orxz,oryx,oryy,oryz,orzx,orzy,orzz;

    tt=t*t; uu=x*x; vv=y*y; ww=z*z;
    vw=y*z; wu=z*x; uv=x*y; ut=x*t; vt=y*t; wt=z*t;
    rxx=tt+uu-vv-ww; rxy=2*(uv-wt); rxz=2*(wu+vt);
    ryy=tt+vv-ww-uu; ryz=2*(vw-ut); ryx=2*(uv+wt);
    rzz=tt+ww-uu-vv; rzx=2*(wu-vt); rzy=2*(vw+ut);
    orxx = o.x.x*rxx+o.x.y*ryx+o.x.z*rzx;
    orxy = o.x.x*rxy+o.x.y*ryy+o.x.z*rzy;
    orxz = o.x.x*rxz+o.x.y*ryz+o.x.z*rzz;
    oryx = o.y.x*rxx+o.y.y*ryx+o.y.z*rzx;
    oryy = o.y.x*rxy+o.y.y*ryy+o.y.z*rzy;
    oryz = o.y.x*rxz+o.y.y*ryz+o.y.z*rzz;
    orzx = o.z.x*rxx+o.z.y*ryx+o.z.z*rzx;
    orzy = o.z.x*rxy+o.z.y*ryy+o.z.z*rzy;
    orzz = o.z.x*rxz+o.z.y*ryz+o.z.z*rzz;
    res.x.x = rxx*orxx+ryx*oryx+rzx*orzx;
    res.x.y = rxx*orxy+ryx*oryy+rzx*orzy;
    res.x.z = rxx*orxz+ryx*oryz+rzx*orzz;
    res.y.x = rxy*orxx+ryy*oryx+rzy*orzx;
    res.y.y = rxy*orxy+ryy*oryy+rzy*orzy;
    res.y.z = rxy*orxz+ryy*oryz+rzy*orzz;
    res.z.x = rxz*orxx+ryz*oryx+rzz*orzx;
    res.z.y = rxz*orxy+ryz*oryy+rzz*orzy;
    res.z.z = rxz*orxz+ryz*oryz+rzz*orzz;
    return res;
}

extern const SPIN SPIN_1 ( VECTOR (1,0,0),0 );
extern const MATRIX MATRIX_0 (VECTOR_0,VECTOR_0,VECTOR_0);

MATRIX:: MATRIX ( ) { }

MATRIX:: MATRIX ( const MATRIX & m)
{
    x = m.x;
	y = m.y;
	z = m.z;
}

MATRIX:: MATRIX ( const VECTOR & X, const VECTOR & Y, const VECTOR & Z)
{
    x = X;
	y = Y;
	z = Z;
}

VECTOR operator | ( const MATRIX & m, const VECTOR & v )
{
    return VECTOR ( m.x|v, m.y|v, m.z|v );
}

MATRIX operator * ( REAL r, const MATRIX & m )
{
    MATRIX m0;
	m0.x = r * m.x;
	m0.y = r * m.y;
	m0.z = r * m.z;
	return m0;
}

REAL det ( const MATRIX & m )
{
    return
        m.x.x*(m.y.y*m.z.z-m.y.z*m.z.y) +
        m.x.y*(m.y.z*m.z.x-m.y.x*m.z.z) +
        m.x.z*(m.y.x*m.z.y-m.y.y*m.z.x);
}
