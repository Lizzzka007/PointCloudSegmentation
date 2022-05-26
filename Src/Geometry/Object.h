#ifndef __OBJECT_H
#define __OBJECT_H

#include  "_defs.h"
#include  "Handle.h"

#pragma pack (push, 4)

typedef void * Arg_list;

#define arg_getRef(args, offset, type) \
        ( (type *)((char *)args + offset) )

#define arg_getValue(args, offset, type) \
        ( *arg_getRef(args, offset, type) )

#define arg_setValue(args, offset, type, value) \
        ( arg_getValue(args, offset, type) = (value) )


struct  INQUARY
{
        int index;
        char * name;
        const char * format;
        const char * protocol;
};

class   STREAM;

class   OBJECT
{
    public:
        int count;
        OBJHNDL * handle_list;

        OBJECT ( );
        OBJECT ( int );
        OBJHNDL * find ( int );
        OBJHNDL * findnext ( int, OBJHNDL * );
        virtual ~ OBJECT ( );
        virtual int  type ( ) const;
        virtual const char * name ( );
        virtual void  rename ( const char * );
        //...
        virtual int     size  ( STREAM * );
        virtual void    read  ( STREAM * );
        virtual void    write ( STREAM * );
        virtual void    awake ( );
        virtual const char * classname ( );
        //...
        friend class OBJHNDL;
};

typedef void ( * obj_func ) ( OBJECT * );

extern void Scene_add   ( obj_func, obj_func, OBJECT *, logical remove = NO );
extern void Contain_add ( obj_func, obj_func, OBJECT * );
extern void execute_add ( OBJECT * );

class   OBJECT_PTR :
public  OBJECT
{
        OBJECT * object;
    public:
        OBJECT_PTR ( OBJECT * );
        OBJECT * getobject ( );
        void    setobject ( OBJECT * );
        int     type () const;
};

extern void company     ( OBJECT & , OBJECT & );
extern void subordinate ( OBJECT & , OBJECT & );

extern int  severance   ( OBJECT & , OBJECT & );

extern OBJHNDL * findgroup ( OBJECT * o, logical (*f) ( OBJECT * ) );
extern OBJHNDL * findgroupnext ( logical (*f) ( OBJECT * ), OBJHNDL * h );
extern OBJHNDL * findobject( OBJECT * o, OBJECT * );

extern logical object_is_diamond ( int );
extern logical object_is_panel   ( int );

extern const char * object_name ( OBJECT * o );
extern const COLOR * object_icon ( OBJECT * o );
extern void  rename_object ( OBJECT * o, const char * );

extern void  destroyobject ( OBJECT * list );

extern logical registerring_new_object ( OBJECT * (*ct) (), int type, const char *, int version = 0 );
extern logical registerring_object_panel ( void (*panel) (OBJECT *), logical (*check) (OBJECT *), const char * title );

extern logical  setToDefaultObject ( const char * key, OBJECT * o );
extern OBJECT * getDefaultObject ( const char * key );

enum OBJECT_TYPES
{
        OBJECT_TYPE =          0x000,
        OBJECT_PTR_TYPE,
        METHOD_LINK_TYPE,
        DIAMOND_TYPE =         0x100,
        CRYSTAL_TYPE =         0x200,
        PANEL_TYPE =           0x400,
        DATA_TYPE =            0x500,
        ICON_TYPE =            0x600,
        FIGURE_PAIR_TYPE =     0x620,
        PHOTO_TYPE =           0x700,
          PHOTO_OBJECT_TYPE,
        SAC_TYPE =             0x800,
        EDIT_LINE_TYPE =       0x900,
        FIGURE_ENTRY_TYPE =    0xA00,
        SYSTEM_PANEL_TYPE =    0xB00,
        EVENT_TYPE =           0xC00,
        LAYER_TYPE =           0xD00,
        GROUP_DEFECT_TYPE =    0xE00,
        DEFECT_ELLIPS_TYPE,
        GROUP_DEFECT_SUPPORT_TYPE = 0xE80,
        CONTOUR_TYPE =         0xF00,
        COUNT_TYPE =          0x1100,
        BASIS_TYPE =          0x1160,
        CHEESE_TYPE =         0x1200,
        STRING_TYPE =         0x1300,
        CARRY_BAG_TYPE =      0x1400,
        CORRECTOR_TYPE =      0x1500,
        LENS_TYPE =           0x1600,
        OBJECT_LIST_TYPE =    0x1700,
        PTR_TITLE_INFO_TYPE = 0x1710,           // removed
        PTR_TITLE_INFO_LIST_TYPE = 0x1711,      // removed
        CUT_STEP_TYPE =       0x1720,
        CUT_PROGRAM_TYPE =    0x1730,
        METHODS_TYPE =        0x1800,
        IB_TYPES =            0x1900,
        MARKING_TYPE =        0x1A00,
        LIGHT_TYPE =          0x1B00,
        OBJECT_BAG_TYPE =     0x1C00,
        CHIP_TYPE =           0x1D00,
        CHIP_BAG_TYPE =       0x1D80,
        SAD_OTHER_TYPE =      0x1E00,
        SEND_EVENT_TYPE =     0x1F00,
        VERSIONS_TYPE =       0x2000,
        NAMES_TYPE, PATH_TYPE, STORAGE_TYPE,
        SCENE_CONTROL_TYPE =  0x2100,
        CONTAINER_DELEGATE_TYPE = 0x2200,           // size = 0x100
        TREE_NODE_TYPES =     0x2300,               // size = 0x100
};

extern logical object_is_f_entry ( OBJECT * );
extern logical object_is_f_diamond ( OBJECT * );
extern logical object_is_f_poly    ( OBJECT * );
extern logical object_is_diamondstorage ( OBJECT * );
extern logical object_is_defect  ( OBJECT * );
extern logical object_is_layer   ( OBJECT * );
extern logical object_is_panel   ( OBJECT * );
extern logical object_is_ib_field( OBJECT * );
extern logical object_is_methods ( OBJECT * );
extern logical object_is_program ( OBJECT * );
extern logical object_is_condition ( OBJECT * );
extern logical object_is_marking ( OBJECT * );
extern logical object_is_light   ( OBJECT * );
extern logical object_is_lens    ( OBJECT * );
extern logical object_is_contour ( OBJECT * );
extern logical object_is_bag     ( OBJECT * );
extern logical object_is_chip    ( OBJECT * );
extern logical object_is_chips   ( OBJECT * );
extern logical object_exist      ( OBJECT * );
extern logical object_is_container_delegate ( OBJECT * );
extern logical object_is_node    ( OBJECT * );

extern OBJECT ALL_FIGURES;
extern OBJECT ALL_OTHER_OBJECTS;

extern void registerForEvent ( int type, OBJECT * o, void (*f) (OBJECT*, OBJECT*) );
extern void unregisterForEvent ( int type, OBJECT * o );
extern void sendEvent ( int type, OBJECT * sender );
extern void SHREDrenewall ( );

enum    SEND_EVENT_TYPES
{
        ACTIVE_WINDOW_DID_CHANGE = SEND_EVENT_TYPE + 1,
        ACTIVE_PHOTO_DID_CHANGE,
        CAMERA_STATE_DID_CHANGE,        // also Light states.
        LINKS_FLAG_DID_CHANGE,
        REFRESH_SCENE,
        ON_IDLE_MESSAGE,
        ON_MOVE_OBJECT,
        ROTATE_SYSTEM_DID_CHANGE,       // motor rotate system
        MARKER_STATE_DID_CHANGE,
        APPRAISER_TABLE_DID_CHANGE,     // изменен набор оценщиков
        DEFAULT_APPRAISER_DID_CHANGE,   // изменен оценщик по умолчанию
        DROP_CONTROLS_DID_CHANGE,       // изменено состояние маркера
        FILE_DID_LOADED,                // был подгружен файл (файлы)
        CLEAR_MEMORY_FOR_WRITE,         // желательно освободить память для записи на диск
        ACTIVE_WINDOW_MAGN_DID_CHANGE,  // изменен коефициент увеличения активного окна
        SYSTEM_REGISTRY_DID_CHANGE,     // изменены системные установки
        BEGIN_INTERSECTING,             // начало пересечения объектов
        END_INTERSECTING,               // окончание пересечения объектов
        DEFECT_WAS_CHANGED,
        CHECK_IN_ALL_OBJECT,            // отследить изменение структуры объекта
        CONTAINER_REFRESH,              // перерисовка сцен изображения объектов
        BEFORE_MOVE_OBJECT,             // перед изменением объекта (sender = BASIS * )
        AFTER_MOVE_OBJECT,              // после изменения объекта (sender = BASIS * )
        ATTACH_TO_ALL_OBJECT,           // добавить объект во все окна
        DETACH_FROM_ALL_OBJECT,         // изъять объект из всех окон
        CONTAINER_ACTION,               // набор действий для сцены (sender = * struct { int mode; ... } )

        SAC_REGISTER_TYPE = SEND_EVENT_TYPE + 0xC0,
};

#define CHANGED 0x1000

#define MODE_READ       1
#define MODE_WRITE      2
#define MODE_CLEAN      4

extern int registerInformationData ( int type, void * data );
extern int getInformationDataLength ( int type );
extern void * getInformationData ( int type, int n );
// Advanced functions
extern void * getInformationDataLast ( int type, int & index,
    logical (*check) (void*, void *), void * context );
extern void * getInformationDataPrev ( int type, int & index,
    logical (*check) (void*, void *), void * context );
extern void * getInformationData ( int type, int & count,
    logical (*check) (void*, void *), void * context );

extern void registerForCheckObject
(
    void (*checkFunc) ( OBJECT & list, OBJECT * o ),
    void (*refreshFunc) ( OBJECT & list )
);

#pragma pack (pop)

#endif  // __OBJECT_H
