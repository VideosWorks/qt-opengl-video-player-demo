/****************************************************************************
** Meta object code from reading C++ file 'sharedglwidget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.8.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../sources/sharedglwidget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'sharedglwidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.8.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_SharedGLWidget_t {
    QByteArrayData data[10];
    char stringdata0[107];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_SharedGLWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_SharedGLWidget_t qt_meta_stringdata_SharedGLWidget = {
    {
QT_MOC_LITERAL(0, 0, 14), // "SharedGLWidget"
QT_MOC_LITERAL(1, 15, 15), // "OnContextWanted"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 20), // "DoVideoFrameReceived"
QT_MOC_LITERAL(4, 53, 8), // "AVFrame*"
QT_MOC_LITERAL(5, 62, 6), // "pFrame"
QT_MOC_LITERAL(6, 69, 15), // "DoContextWanted"
QT_MOC_LITERAL(7, 85, 5), // "CHECK"
QT_MOC_LITERAL(8, 91, 8), // "CUresult"
QT_MOC_LITERAL(9, 100, 6) // "result"

    },
    "SharedGLWidget\0OnContextWanted\0\0"
    "DoVideoFrameReceived\0AVFrame*\0pFrame\0"
    "DoContextWanted\0CHECK\0CUresult\0result"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_SharedGLWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   34,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       3,    1,   35,    2, 0x09 /* Protected */,
       6,    0,   38,    2, 0x09 /* Protected */,
       7,    1,   39,    2, 0x09 /* Protected */,

 // signals: parameters
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 4,    5,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 8,    9,

       0        // eod
};

void SharedGLWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        SharedGLWidget *_t = static_cast<SharedGLWidget *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->OnContextWanted(); break;
        case 1: _t->DoVideoFrameReceived((*reinterpret_cast< AVFrame*(*)>(_a[1]))); break;
        case 2: _t->DoContextWanted(); break;
        case 3: _t->CHECK((*reinterpret_cast< CUresult(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (SharedGLWidget::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&SharedGLWidget::OnContextWanted)) {
                *result = 0;
                return;
            }
        }
    }
}

const QMetaObject SharedGLWidget::staticMetaObject = {
    { &QOpenGLWidget::staticMetaObject, qt_meta_stringdata_SharedGLWidget.data,
      qt_meta_data_SharedGLWidget,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *SharedGLWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *SharedGLWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_SharedGLWidget.stringdata0))
        return static_cast<void*>(const_cast< SharedGLWidget*>(this));
    if (!strcmp(_clname, "QOpenGLFunctions"))
        return static_cast< QOpenGLFunctions*>(const_cast< SharedGLWidget*>(this));
    return QOpenGLWidget::qt_metacast(_clname);
}

int SharedGLWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QOpenGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 4)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 4;
    }
    return _id;
}

// SIGNAL 0
void SharedGLWidget::OnContextWanted()
{
    QMetaObject::activate(this, &staticMetaObject, 0, Q_NULLPTR);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
