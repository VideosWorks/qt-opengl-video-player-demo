/****************************************************************************
** Meta object code from reading C++ file 'glwidget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.8.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../sources/glwidget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'glwidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.8.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_GLWidget_t {
    QByteArrayData data[15];
    char stringdata0[167];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_GLWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_GLWidget_t qt_meta_stringdata_GLWidget = {
    {
QT_MOC_LITERAL(0, 0, 8), // "GLWidget"
QT_MOC_LITERAL(1, 9, 7), // "clicked"
QT_MOC_LITERAL(2, 17, 0), // ""
QT_MOC_LITERAL(3, 18, 15), // "OnContextWanted"
QT_MOC_LITERAL(4, 34, 20), // "DoVideoFrameReceived"
QT_MOC_LITERAL(5, 55, 8), // "AVFrame*"
QT_MOC_LITERAL(6, 64, 6), // "pFrame"
QT_MOC_LITERAL(7, 71, 16), // "onAboutToCompose"
QT_MOC_LITERAL(8, 88, 14), // "onFrameSwapped"
QT_MOC_LITERAL(9, 103, 15), // "onAboutToResize"
QT_MOC_LITERAL(10, 119, 9), // "onResized"
QT_MOC_LITERAL(11, 129, 15), // "DoContextWanted"
QT_MOC_LITERAL(12, 145, 5), // "CHECK"
QT_MOC_LITERAL(13, 151, 8), // "CUresult"
QT_MOC_LITERAL(14, 160, 6) // "result"

    },
    "GLWidget\0clicked\0\0OnContextWanted\0"
    "DoVideoFrameReceived\0AVFrame*\0pFrame\0"
    "onAboutToCompose\0onFrameSwapped\0"
    "onAboutToResize\0onResized\0DoContextWanted\0"
    "CHECK\0CUresult\0result"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_GLWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   59,    2, 0x06 /* Public */,
       3,    0,   60,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       4,    1,   61,    2, 0x09 /* Protected */,
       7,    0,   64,    2, 0x09 /* Protected */,
       8,    0,   65,    2, 0x09 /* Protected */,
       9,    0,   66,    2, 0x09 /* Protected */,
      10,    0,   67,    2, 0x09 /* Protected */,
      11,    0,   68,    2, 0x09 /* Protected */,
      12,    1,   69,    2, 0x09 /* Protected */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 5,    6,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 13,   14,

       0        // eod
};

void GLWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        GLWidget *_t = static_cast<GLWidget *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->clicked(); break;
        case 1: _t->OnContextWanted(); break;
        case 2: _t->DoVideoFrameReceived((*reinterpret_cast< AVFrame*(*)>(_a[1]))); break;
        case 3: _t->onAboutToCompose(); break;
        case 4: _t->onFrameSwapped(); break;
        case 5: _t->onAboutToResize(); break;
        case 6: _t->onResized(); break;
        case 7: _t->DoContextWanted(); break;
        case 8: _t->CHECK((*reinterpret_cast< CUresult(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (GLWidget::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&GLWidget::clicked)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (GLWidget::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&GLWidget::OnContextWanted)) {
                *result = 1;
                return;
            }
        }
    }
}

const QMetaObject GLWidget::staticMetaObject = {
    { &QOpenGLWidget::staticMetaObject, qt_meta_stringdata_GLWidget.data,
      qt_meta_data_GLWidget,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *GLWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *GLWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_GLWidget.stringdata0))
        return static_cast<void*>(const_cast< GLWidget*>(this));
    if (!strcmp(_clname, "QOpenGLExtraFunctions"))
        return static_cast< QOpenGLExtraFunctions*>(const_cast< GLWidget*>(this));
    return QOpenGLWidget::qt_metacast(_clname);
}

int GLWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QOpenGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 9)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 9;
    }
    return _id;
}

// SIGNAL 0
void GLWidget::clicked()
{
    QMetaObject::activate(this, &staticMetaObject, 0, Q_NULLPTR);
}

// SIGNAL 1
void GLWidget::OnContextWanted()
{
    QMetaObject::activate(this, &staticMetaObject, 1, Q_NULLPTR);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
