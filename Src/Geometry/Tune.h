#ifndef __TUNE_H__
#define __TUNE_H__

#pragma pack (push, 4)

// #define M_PI            3.14159265358979323846
// #define M_SQRT2         1.41421356237309504880

inline int    qmod ( const int    & v ) { return v * v; }
inline double qmod ( const double & v ) { return v * v; }

inline int round ( const double & x )
{
    return int ( x < 0 ? x - 0.5 : x + 0.5 );
}

#pragma pack (pop)

#endif //__TUNE_H__
