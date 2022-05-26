/*  STREAM.H

    I/O stream.

    Copyright (c) 1994 by Octopus Ltd.
    All rights reserved.
*/

#ifndef __STREAM_H
#define __STREAM_H

// #include <io.h>

#pragma pack (push, 4)

class   OBJECT;

class   STREAM
{
public:
	//      for read:
	virtual void    read ( void * , uint ) = 0;
	virtual int     readobjectref ( void ** ) = 0;

	// Returns a pointer to its internal buffer
	// Copy the string from this buffer before the next call to readstring()
	// Returns NULL pointer if the written string was a NULL pointer or an empty string
	virtual const char * readstring() = 0;
	//      for write:
	virtual void    write ( const void * , uint ) = 0;
	virtual int     writeobjectref ( const void * ) = 0;
	virtual void    writestring ( const char * ) = 0;
	//      for size:
	virtual int     sizeobjectref ( ) = 0;
	virtual int     sizestring ( const char * ) = 0;

	/*
	Функции VersionControl предназначены для контроля записи версий объектов,
	так чтобы, более ранние версии программы могли читать более поздние версии
	данных. Более поздние данные должны записываться первыми. В этом случае
	функция read осуществит необходимый сдвиг.
	*/
	// Контроль версий. Размер памяти под контроль версий.
	int VersionControlSize(int nProgramVersion) const;
	// Контроль версий. Запись блока.
	void VersionControlWrite(int nProgramVersion, const int * pNDataVersionSize);
	// Контроль версий. Чтение блока.
	void VersionControlRead(int & nDataVersion, int nProgamVersion);
};

typedef struct tagLINK_INFO { int type, lnum; } LINK_INFO;

typedef struct tagTITLE_INFO
{
        int offset;
        int lnum;
        int type;
        int size;
        int nlink;
        LINK_INFO * links;
        OBJECT * object;
}
        TITLE_INFO;

class   ARCHIVING_STREAM :
public  STREAM
{
	int file;
	int total;
	TITLE_INFO * info;
public:
	ARCHIVING_STREAM ( int handle );
	void    settitle ( TITLE_INFO *, int );
	void    seek ( int offset, int whence );
	int     tell ( );
	void    read ( void * , uint );
	int     readobjectref ( void ** );
	const char * readstring();
	void    write ( const void * , uint );
	int     writeobjectref ( const void * );
	void    writestring ( const char * );
	int     sizeobjectref ( );
	int     sizestring ( const char * );
};

#pragma pack (pop)

#endif
