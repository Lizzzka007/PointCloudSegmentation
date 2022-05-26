#ifndef __OBJHNDL_H
#define __OBJHNDL_H

#include  "_defs.h"

#pragma pack (push, 4)

class   OBJECT;

enum    OBJHNDL_TYPES
{
        EQUAL = 1,
        PARENT,
        CHILD,
        OF_SLAVE
};


class   OBJHNDL
{
    private:
        OBJHNDL * _h1, * _h2;
        OBJHNDL * _companion;
        OBJECT & _object;
    public:
        OBJHNDL ( OBJECT & );
        virtual ~ OBJHNDL ( );
    public:
        OBJECT * object ( );
        OBJHNDL * next ( ) const;
        virtual int type ( ) const;
        friend class OBJECT;
        friend class OBJHNDL_OF_SLAVE;
        friend void company ( OBJECT & a, OBJECT & b );
        friend void subordinate ( OBJECT & a, OBJECT & b );
        friend void make_depend ( OBJECT & Master, OBJECT & Slave );
};

class   OBJHNDL_CHILD :
public  OBJHNDL
{
        OBJHNDL_CHILD ( OBJECT & );
        int type ( ) const;
        friend void subordinate ( OBJECT & a, OBJECT & b );
};

class   OBJHNDL_PARENT :
public  OBJHNDL
{
        OBJHNDL_PARENT ( OBJECT & );
        int type ( ) const;
        friend void subordinate ( OBJECT & a, OBJECT & b );
};

#pragma pack (pop)

#endif  //  __OBJHNDL_H
