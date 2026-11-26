/* -*-c++-*-
 *  ----------------------------------------------------------------------------
 *
 *       PlantGL: The Plant Graphic Library
 *
 *       Copyright CIRAD/INRIA/INRA
 *
 *       File author(s): F. Boudon (frederic.boudon@cirad.fr) et al. 
 *
 *  ----------------------------------------------------------------------------
 *
 *   This software is governed by the CeCILL-C license under French law and
 *   abiding by the rules of distribution of free software.  You can  use, 
 *   modify and/ or redistribute the software under the terms of the CeCILL-C
 *   license as circulated by CEA, CNRS and INRIA at the following URL
 *   "http://www.cecill.info". 
 *
 *   As a counterpart to the access to the source code and  rights to copy,
 *   modify and redistribute granted by the license, users are provided only
 *   with a limited warranty  and the software's author,  the holder of the
 *   economic rights,  and the successive licensors  have only  limited
 *   liability. 
 *       
 *   In this respect, the user's attention is drawn to the risks associated
 *   with loading,  using,  modifying and/or developing or reproducing the
 *   software by the user in light of its specific status of free software,
 *   that may mean  that it is complicated to manipulate,  and  that  also
 *   therefore means  that it is reserved for developers  and  experienced
 *   professionals having in-depth computer knowledge. Users are therefore
 *   encouraged to load and test the software's suitability as regards their
 *   requirements in conditions enabling the security of their systems and/or 
 *   data to be ensured and,  more generally, to use and operate it in the 
 *   same conditions as regards security. 
 *
 *   The fact that you are presently reading this means that you have had
 *   knowledge of the CeCILL-C license and that you accept its terms.
 *
 *  ----------------------------------------------------------------------------
 */


#include "geomscenegl.h"

/// Tools
#include <plantgl/tool/util_string.h>
#include <plantgl/tool/bfstream.h>

/// GEOM
#include <plantgl/scenegraph/scene/shape.h>
#include <plantgl/scenegraph/scene/factory.h>
#include <plantgl/scenegraph/geometry/amapsymbol.h>

/// Action
#include <plantgl/algo/base/polygoncomputer.h>
#include <plantgl/algo/codec/povprinter.h>
#include <plantgl/algo/codec/plyprinter.h>
#include <plantgl/algo/codec/vrmlprinter.h>
#include <plantgl/algo/codec/vgstarprinter.h>
#include <plantgl/algo/codec/binaryprinter.h>
#include <plantgl/algo/codec/linetreeprinter.h>
#include <plantgl/algo/codec/ligfile.h>
#include <plantgl/algo/codec/vgsfile.h>
#include <plantgl/algo/codec/geomview_file.h>

/// Viewer
#include "../base/light.h"
#include "../base/camera.h"
#include "../base/icons.h"
#include "../base/glframe.h"
#include "../base/filemanager.h"


#include "linetree.h"
#include "reader.h"
#include "interface/ui_importselection.h"

/// Qt
#include <QtGlobal>
#include <QTimer>
#include <functional>
#if QT_VERSION >= QT_VERSION_CHECK(5,0,0)
    #include <QtWidgets/qlabel.h>
    #include <QtWidgets/qcombobox.h>
    #include <QtWidgets/qgroupbox.h>
    #include <QtWidgets/qmessagebox.h>
    #include <QtWidgets/qfiledialog.h>
    #include <QtWidgets/qslider.h>
    #include <QtWidgets/qpushbutton.h>
    #include <QtWidgets/qradiobutton.h>
    #include <QtWidgets/qapplication.h>
#else
    #include <QtGui/qlabel.h>
    #include <QtGui/qcombobox.h>
    #include <QtGui/qgroupbox.h>
    #include <QtGui/qmessagebox.h>
    #include <QtGui/qfiledialog.h>
    #include <QtGui/qslider.h>
    #include <QtGui/qpushbutton.h>
    #include <QtGui/qradiobutton.h>
    #include <QtGui/qapplication.h>
#endif


#ifdef QT_THREAD_SUPPORT
#ifndef PGL_DEBUG
#define GEOM_THREAD
#endif
#endif

PGL_USING_NAMESPACE
using namespace std;
/* ----------------------------------------------------------------------- */

bool
ViewGeomSceneGL::addOpenEntries(QMenu * menu)
{
  QPixmap openIcon( ViewerIcon::getPixmap(ViewerIcon::fileopen ));
  menu->addAction( openIcon, tr("Open &Geom File"),
                       this,SLOT(openGeomFile()),Qt::CTRL+Qt::Key_G);
  menu->addAction( openIcon, tr("&Add Geom File"),
                       this,SLOT(addGeomFile()));
  return true;
}

bool
ViewGeomSceneGL::addSaveEntries(QMenu * menu)
{
  QPixmap saveIcon(ViewerIcon::getPixmap(ViewerIcon::filefloppy ));
  menu->addAction( saveIcon, tr("Save As &Geom"),
                       this,SLOT(saveAsGeom()));
  menu->addAction( saveIcon, tr("Save &Selection"),
                       this,SLOT(saveSelection()));
  menu->addAction( saveIcon, tr("Save &Not Selection"),
                       this,SLOT(saveNotSelection()));
  return true;
}

bool
ViewGeomSceneGL::addImportEntries(QMenu * menu)
{
  QPixmap openIcon(ViewerIcon::getPixmap(ViewerIcon::fileopen ));
  menu->addAction( openIcon , tr("Import &AmapSymbol"),
                    this,SLOT(openAmapSymbol()),Qt::CTRL+Qt::Key_A);
  menu->addAction( openIcon, tr("Import &Linetree Files"),
                       this,SLOT(openLinetree()),Qt::CTRL+Qt::Key_L);
  menu->addAction( openIcon, tr("Import &GeomView Files"),
                       this,SLOT(openGeomViewFile()));
  menu->addAction( openIcon, tr("Import &VegeStar Files"),
                       this,SLOT(openVegeStar()));
  QMenu * submenu = new QMenu(menu);
  submenu->addAction( openIcon , tr("Import Symbol")+" 11",
                    this,SLOT(openVegeStarSymbol1()));
  submenu->addAction( openIcon , tr("Import Symbol")+" 12",
                    this,SLOT(openVegeStarSymbol2()));
  submenu->addAction( openIcon , tr("Import Symbol")+" 13",
                    this,SLOT(openVegeStarSymbol3()));
  submenu->addSeparator();
  submenu->addAction( openIcon , tr("Show Symbol")+" 11",
                    this,SLOT(showVegeStarSymbol1()));
  submenu->addAction( openIcon , tr("Show Symbol")+" 12",
                    this,SLOT(showVegeStarSymbol2()));
  submenu->addAction( openIcon , tr("Show Symbol")+" 13",
                    this,SLOT(showVegeStarSymbol3()));
  submenu->addSeparator();
  submenu->addAction( openIcon , tr("Set Selection as Symbol")+" 11",
                    this,SLOT(setSelectionVegeStarSymbol1()));
  submenu->addAction( openIcon , tr("Set Selection as Symbol")+" 12",
                    this,SLOT(setSelectionVegeStarSymbol1()));
  submenu->addAction( openIcon , tr("Set Selection as Symbol")+" 13",
                    this,SLOT(setSelectionVegeStarSymbol1()));
  submenu->setTitle(tr("Import VegeStar Symbol"));
  menu->addMenu(submenu);
  return true;
}

bool
ViewGeomSceneGL::addExportEntries(QMenu * menu)
{
  QPixmap saveIcon( ViewerIcon::getPixmap(ViewerIcon::filefloppy) );
  QPixmap savePovIcon( ViewerIcon::getPixmap( ViewerIcon::pov) );
  QPixmap saveVrmlIcon( ViewerIcon::getPixmap(ViewerIcon::vrml) );
  menu->addAction( saveIcon , tr("Export as &AmapSymbol"),
                    this,SLOT(saveAsAmapSymbol()));
  menu->addAction( saveIcon , tr("Export as &Linetree"),
                    this,SLOT(saveAsLinetree()));
  menu->addAction( savePovIcon , tr("Export as Pov&Ray"),
                    this,SLOT(saveAsPovRay()),Qt::CTRL+Qt::Key_R);
  menu->addAction( saveVrmlIcon , tr("Export as Vr&ml"),
                    this,SLOT(saveAsVrml()),Qt::CTRL+Qt::Key_M);
  menu->addAction( saveIcon , tr("Export as Ply"),
                    this,SLOT(saveAsPly()));
  menu->addAction( saveIcon , tr("Export as V&egeStar"),
                    this,SLOT(saveAsVegeStar()),Qt::CTRL+Qt::Key_E);
  return true;
}

/* ----------------------------------------------------------------------- */
void
ViewGeomSceneGL::addFile(const QString& filename)
{
  if(!filename.isEmpty()){
    QString ext = QFileInfo(filename).suffix();
    ext = ext.toUpper();
    if(ext == "SMB"){
      openAmapSymbol(filename,true);
    }
    else if(ext == "GEO" || ext == "GEOM" || ext == "BGEOM" ){
      addGeomFile(filename);
    }
    else if(ext == "LIG"){
      openLinetree(filename,true);
    }
    else if(ext == "VGX" || ext == "VGS"){
      openVegeStar(filename,true);
    }
    else if(ext == "POL"){
      openVegeStarSymbol1(filename,true);
    }
    else if(ext == "QUAD" || ext == "BEZ" || ext == "OFF"){
      openGeomViewFile(filename);
    }
  }
}

bool
ViewGeomSceneGL::open(const QString& filename)
{
  if(!filename.isEmpty()){
    QString ext = QFileInfo(filename).suffix();
    ext = ext.toUpper();
    if(ext == "SMB"){
      openAmapSymbol(filename);
      return true;
    }
    else if(ext == "GEO" || ext == "GEOM" || ext == "BGEOM" ){
      return openGeomFile(filename);
    }
    else if(ext == "LIG"){
      return openLinetree(filename);
    }
    else if(ext == "VGX" || ext == "VGS"){
      return openVegeStar(filename);
    }
    else if(ext == "POL"){
      return openVegeStarSymbol1(filename);
    }
    else if(ext == "QUAD" || ext == "BEZ" || ext == "OFF"){
      return openGeomViewFile(filename);
    }
    else if(SceneFactory::get().isReadable(filename.toStdString())){
      return openGeomFile(filename);
    }
    else {
            static int selection = 5;
            QDialog dialog(__frame);
            dialog.setModal(true);
            Ui::ViewImporterSelection importer;
            importer.setupUi(&dialog);
            QObject::connect(importer.OkButton,SIGNAL(pressed()),&dialog,SLOT(accept()));
            QObject::connect(importer.CancelButton,SIGNAL(pressed()),&dialog,SLOT(reject()));
            importer.Type->setText(ext);
            importer.File->setText(QFileInfo(filename).fileName());
            importer.ImporterCombo->setCurrentIndex(selection);
            if(dialog.exec() == QDialog::Rejected)return false;
            else {
                selection = importer.ImporterCombo->currentIndex();
                switch(selection){
                case 0 :return openGeomFile(filename);break;
                case 1 :return openAmapSymbol(filename);break;
                case 2 :return openLinetree(filename);break;
                case 3 :return openVegeStar(filename);break;
                case 4 :return openVegeStarSymbol1(filename);break;
                case 5 :return openGeomViewFile(filename);break;
                default : return false; break;
                }
            }
        }
  }
  else {
            QMessageBox::warning(__frame,tr("Empty Filename"),
                         tr("Cannot open Empty filename"),1,0,0);
          return false;
  }
}
/*
bool
ViewGeomSceneGL::openStream(std::istream& stream)
{
#ifdef GEOM_THREAD
  if(__reader && __reader->isRunning()){
    QMessageBox::warning(__frame,tr("Already Reading File"),
                         QString(tr("Currently Reading File")+" %1").
                         arg(__reader->getFilename()),1,0,0);
    return false;
  }
  else{
    if(__reader) {
      delete __reader;
      __reader = NULL;
    }
    __reader = new ViewGeomStreamReader(&stream,this);
    __reader->start();
    return true;
  }
#else
    stringstream _errlog(ios::out) ;
    ScenePtr scene = ScenePtr(new Scene(stream,_errlog));
    setScene(scene);
    string _msg = _errlog.str();
    if(!_msg.empty())error(_msg.c_str());
    emit valueChanged();
    setFilename("");
    return true;
#endif
  return false;
}*/

/* ----------------------------------------------------------------------- */

namespace {

void showFileDialog(QDialog * dialog)
{
  // Let the menu close before the native panel takes keyboard focus.
  dialog->setWindowModality(Qt::WindowModal);
  dialog->setAttribute(Qt::WA_DeleteOnClose);
  QTimer::singleShot(0, dialog, [dialog]() {
    dialog->show();
    dialog->activateWindow();
  });
}

void selectExistingFile(QWidget * frame, QObject * context, const QString& caption,
                        const QString& initial, const QString& filter,
                        const std::function<void(const QString&)>& onSelected)
{
  auto * dialog = new QFileDialog(frame->window(), caption, initial, filter);
  dialog->setFileMode(QFileDialog::ExistingFile);
  QObject::connect(dialog, &QFileDialog::fileSelected, context, onSelected, Qt::QueuedConnection);
  showFileDialog(dialog);
}

void selectSaveFile(QWidget * frame, QObject * context, const QString& initial,
                    const QString& extension, const QString& filter,
                    const std::function<void(const QString&)>& onSelected)
{
  auto * dialog = new QFileDialog(frame->window(), QStringLiteral("Save"),
                                 QFileInfo(initial).path(), filter);
  dialog->setAcceptMode(QFileDialog::AcceptSave);
  dialog->setDefaultSuffix(extension);
  QObject::connect(dialog, &QFileDialog::fileSelected, context,
                   [frame, extension, onSelected](const QString& filename) {
    const QFileInfo file(filename);
    if (!extension.isEmpty() && extension.compare(file.suffix(), Qt::CaseInsensitive) != 0) {
      const QString target = file.dir().filePath(file.completeBaseName() + '.' + extension);
      if (QFile::exists(target) &&
          QMessageBox::warning(frame->window(), QObject::tr("File Already Exists"),
              QObject::tr("%1 already exists. Overwrite?").arg(target),
              QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel) != QMessageBox::Yes) {
        return;
      }
      onSelected(target);
    }
    else {
      onSelected(filename);
    }
  }, Qt::QueuedConnection);
  showFileDialog(dialog);
}

} // namespace

void
ViewGeomSceneGL::openGeomFile()
{
  selectExistingFile(__frame, this, "Open PlantGL", getFilename(), tr("PlantGL File")+" (*.geom;*.bgeom);;"+tr("All Files (*.*)"),
                     [this](const QString& filename) {
    openGeomFile(filename);
  });
}

void
ViewGeomSceneGL::addGeomFile()
{
  selectExistingFile(__frame, this, "Open PlantGL", getFilename(), tr("PlantGL File")+" (*.geom;*.bgeom);;"+tr("All Files (*.*)"),
                     [this](const QString& filename) {
    addGeomFile(filename);
  });
}

/* ----------------------------------------------------------------------- */

void
ViewGeomSceneGL::openAmapSymbol()
{
  selectExistingFile(__frame, this, "Open Symbol", getFilename(), tr("Amap Symbol")+" (*.smb);;"+tr("All Files (*.*)"),
                     [this](const QString& filename) {
    openAmapSymbol(filename);
  });
}



void
ViewGeomSceneGL::openGeomViewFile()
{
  selectExistingFile(__frame, this, "Open GeomView File", getFilename(), "",
                     [this](const QString& filename) {
    openGeomViewFile(filename);
  });
}

void
ViewGeomSceneGL::openLinetree()
{
  openLinetree(QString());
}


void
ViewGeomSceneGL::openVegeStar()
{
  selectExistingFile(__frame, this, "Open VegeStar File", getFilename(), tr("VegeStar File")+" (*.vgx;*.vgs);;"+tr("All Files (*.*)"),
                     [this](const QString& filename) {
    openVegeStar(filename);
  });
}

void
ViewGeomSceneGL::openVegeStarSymbol1()
{
  selectExistingFile(__frame, this, "Open Symbol", getFilename(), tr("VegeStar Symbol")+" (*.pol);;"+tr("All Files (*.*)"),
                     [this](const QString& filename) {
    openVegeStarSymbol1(filename);
  });
}

bool
ViewGeomSceneGL::openVegeStarSymbol1(const QString& filename,bool add)
{
  if(!filename.isEmpty()){
    stringstream _errlog(ios::out) ;
    GeometryPtr _symbol = VegeStarFile::importPolygonFile(filename.toStdString(),_errlog);
    string _msg = _errlog.str();
    if(!_msg.empty())error(_msg.c_str());

    if (_symbol->isValid()) {
      VegeStarFile::setShape11(_symbol);
      ScenePtr scene(new Scene());
      scene->add(Shape3DPtr(new Shape(_symbol)));
      if(add){
        addScene(scene);
        return true;
      }
      else {
      if(setScene(scene) ==1){
        emit valueChanged();
        setFilename(filename);
        return true;
      }
      else return false;
      }
    }
    else return false;
  }
  else return false;
}

void
ViewGeomSceneGL::openVegeStarSymbol2()
{
  selectExistingFile(__frame, this, "Open Symbol", getFilename(), tr("VegeStar Symbol")+" (*.pol);;"+tr("All Files (*.*)"),
                     [this](const QString& filename) {
    stringstream _errlog(ios::out) ;
    GeometryPtr _symbol = VegeStarFile::importPolygonFile(filename.toStdString(),_errlog);
    string _msg = _errlog.str();
    if(!_msg.empty())error(_msg.c_str());

    if (_symbol->isValid()) {
      VegeStarFile::setShape12(_symbol);
      ScenePtr scene(new Scene());
      scene->add(Shape3DPtr(new Shape(_symbol)));

      if(setScene(scene) ==1){
        emit valueChanged();
        setFilename(filename);
      }
    }
  });
}

void
ViewGeomSceneGL::openVegeStarSymbol3()
{
  selectExistingFile(__frame, this, "Open Symbol", getFilename(), tr("VegeStar Symbol")+" (*.pol);;"+tr("All Files (*.*)"),
                     [this](const QString& filename) {
    stringstream _errlog(ios::out) ;
    GeometryPtr _symbol = VegeStarFile::importPolygonFile(filename.toStdString(),_errlog);
    string _msg = _errlog.str();
    if(!_msg.empty())error(_msg.c_str());

    if (_symbol->isValid()) {
      VegeStarFile::setShape13(_symbol);
      ScenePtr scene(new Scene());
      scene->add(Shape3DPtr(new Shape(_symbol)));

      if(setScene(scene) ==1){
        emit valueChanged();
        setFilename(filename);
      }
    }
  });
}
void
ViewGeomSceneGL::showVegeStarSymbol1(){
      GeometryPtr _symbol = VegeStarFile::getShape11();
      ScenePtr scene(new Scene());
      scene->add(Shape3DPtr(new Shape(_symbol)));

      if(setScene(scene) ==1){
        emit valueChanged();
        setFilename("");
      }
 }

void
ViewGeomSceneGL::setSelectionVegeStarSymbol1(){
  if(!__selectedShapes.empty()){
    ShapePtr sh = dynamic_pointer_cast<Shape>(get_item_value(__selectedShapes.begin()));
    if(sh&&sh->getGeometry())
      VegeStarFile::setShape11(sh->getGeometry());
  }
 }

void
ViewGeomSceneGL::setSelectionVegeStarSymbol2(){
  if(!__selectedShapes.empty()){
    ShapePtr sh = dynamic_pointer_cast<Shape>(get_item_value(__selectedShapes.begin()));
    if(sh&&sh->getGeometry())
      VegeStarFile::setShape12(sh->getGeometry());
  }
 }

void
ViewGeomSceneGL::setSelectionVegeStarSymbol3(){
  if(!__selectedShapes.empty()){
    ShapePtr sh = dynamic_pointer_cast<Shape>(get_item_value(__selectedShapes.begin()));
    if(sh&&sh->getGeometry())
      VegeStarFile::setShape13(sh->getGeometry());
  }
 }

void
ViewGeomSceneGL::showVegeStarSymbol2(){
      GeometryPtr _symbol = VegeStarFile::getShape12();
      ScenePtr scene(new Scene());
      scene->add(Shape3DPtr(new Shape(_symbol)));

      if(setScene(scene) ==1){
        emit valueChanged();
        setFilename("");
      }
 }

void
ViewGeomSceneGL::showVegeStarSymbol3(){
      GeometryPtr _symbol = VegeStarFile::getShape13();
      ScenePtr scene(new Scene());
      scene->add(Shape3DPtr(new Shape(_symbol)));

      if(setScene(scene) ==1){
        emit valueChanged();
        setFilename("");
      }
 }

/* ----------------------------------------------------------------------- */

bool
ViewGeomSceneGL::openGeomFile(const QString& filename)
{
  if(!filename.isEmpty()){
#ifdef GEOM_THREAD
    if(__reader && __reader->isRunning()){
      QMessageBox::warning(__frame,tr("Already Reading File"),
                           QString(tr("Currently Reading File")+" %1").
                           arg(__reader->getFilename()),1,0,0);
      return false;
    }
    else {
      if(__reader) {
        delete __reader;
        __reader = NULL;
      }
      __reader = new ViewGeomReader(filename,
                                    this);
      __reader->start();
      return true;
    }
#else
    stringstream _errlog(ios::out) ;
    ScenePtr scene = ScenePtr(new Scene(filename.toStdString(),"",_errlog));
    setScene(scene);
    string _msg = _errlog.str();
    if(!_msg.empty())error(_msg.c_str());
    emit valueChanged();
    setFilename(filename);
    return true;
#endif
  }
  return false;
}

bool
ViewGeomSceneGL::addGeomFile(const QString& filename)
{
  if(!filename.isEmpty()){
#ifdef GEOM_THREAD
    if(__reader && __reader->isRunning()){
      __reader->wait();
    }
    if(__reader) {
      delete __reader;
      __reader = NULL;
    }
    __reader = new ViewGeomReader(filename,
      this,-1,true);
    __reader->start();
    return true;
#else
    stringstream _errlog(ios::out) ;
    ScenePtr scene = ScenePtr(new Scene(filename.toStdString(),"",_errlog));
    addScene(scene);
    string _msg = _errlog.str();
    if(!_msg.empty())error(_msg.c_str());
    else error(_msg.c_str());
    emit valueChanged();
    return true;
#endif
  }
  return false;
}

/* ----------------------------------------------------------------------- */

bool
ViewGeomSceneGL::openAmapSymbol(const QString& filename,bool add)
{
  if(! filename.isEmpty()) {
    AppearancePtr _material(new Material());
    _material->setName(string("Default"));
    GeometryPtr _symbol;
    stringstream _errlog(ios::out) ;
    {
        PglErrorStream::Binder psb(_errlog);
        _symbol = GeometryPtr(new AmapSymbol(filename.toStdString(),
                               AmapSymbol::DEFAULT_SOLID));
    }
    string _msg = _errlog.str();
    if(!_msg.empty())error(_msg.c_str());

    if (_symbol->isValid()) {

      ScenePtr scene(new Scene());
      scene->add(Shape3DPtr(new Shape(_symbol,_material)));
      if(add){
        addScene(scene);
        return true;
      }
      else {
      if(setScene(scene) ==1){
          setFilename(filename);

          return true;
      }
      else return false;
      }
    }
    else return false;
  }
  else return false;
}



bool
ViewGeomSceneGL::openGeomViewFile(const QString& filename,bool add)
{
  if(!filename.isEmpty()){
    stringstream _errlog(ios::out) ;
    ScenePtr scene = GeomViewBuilder::Parse(filename.toStdString(),_errlog);
    if(add)addScene(scene);
    else {
      setScene(scene);
      setFilename(filename);
    }
    string _msg = _errlog.str();
    if(!_msg.empty() )error(_msg.c_str());
    emit valueChanged();
    return true;
  }
  return false;
}


bool
ViewGeomSceneGL::openLinetree(const QString& filename,bool add)
{
  ViewReadLinetree __linetreeDialog;
  if(!filename.isEmpty())__linetreeDialog.setLigFile(filename);
  if(__linetreeDialog.exec()){
#ifdef GEOM_THREAD
    if(__reader && __reader->isRunning()){
      if(!add){
      QMessageBox::warning(__frame,tr("Already Reading File"),
                           QString(tr("Currently Reading File")+" %1").
                           arg(__reader->getFilename()),1,0,0);
      return false;
      }
      else __reader->wait();
    }
    else{
      if(__reader) {
        delete __reader;
        __reader = NULL;
      }
      __reader = new ViewLinetreeReader(__linetreeDialog.getLigFile(),
                                    __linetreeDialog.getDtaFile(),
                                    __linetreeDialog.getSMBPath(),
                                    this,
                                    __linetreeDialog.bigEndian(),add);
      __reader->start();
      return true;
    }
#else
    ScenePtr scene = readLineTree(string(__linetreeDialog.getLigFile().toStdString()),
                                  string(__linetreeDialog.getDtaFile().toStdString()),
                                  string(__linetreeDialog.getSMBPath().toStdString()),
                                  __linetreeDialog.bigEndian(),cerr);
    if(add){
      addScene(scene);
      return true;
    }
    else {
      if(setScene(scene) == 1)
      {
        setFilename(__linetreeDialog.getLigFile());
        return true;
      }
      else return false;
    }
#endif
  }
  return false;
}

bool
ViewGeomSceneGL::openVegeStar(const QString& filename,bool add)
{
  if(!filename.isEmpty()){
/*#ifndef QT_THREAD_SUPPORT
    if(__reader && __reader->isRunning()){
      QMessageBox::warning(__frame,"Already Reading File",
                           QString("Reading Yet File %1").
                           arg(__reader->getFilename()),1,0,0);
      return false;
    }
    else {
      if(__reader) {
        delete __reader;
        __reader = NULL;
      }
      __reader = new ViewVegeStarReader(filename,
                                        this);
      __reader->start();
      return true;
    }
#else*/
    stringstream _errlog(ios::out) ;
    ScenePtr scene = VegeStarFile::read(filename.toStdString(),_errlog);
    if(add)addScene(scene);
    else {
      setScene(scene);
      setFilename(filename);
    }
    string _msg = _errlog.str();
    if(!_msg.empty())error(_msg.c_str());
    emit valueChanged();
    return true;
// #endif
  }
  return false;
}

/* ----------------------------------------------------------------------- */
void
ViewGeomSceneGL::save()
{
  if(getFilename().isEmpty()){
    saveAs();return; }
  QString extension=QFileInfo(getFilename()).suffix();
  extension= extension.toUpper();
  if(extension.isEmpty())saveAs();
  else if(extension == "GEOM" || extension == "BGEOM" )
    saveAsGeom(getFilename());
  else if(extension == "VGX" )
    saveAsVegeStar(getFilename());
  else if(extension == "LIG" )
    saveAsLinetree(getFilename());
  else saveAs();
  setFilename(getFilename());
}

void
ViewGeomSceneGL::saveAs()
{
  selectSaveFile(__frame, this, getFilename(), QString(),
                 tr("PlantGL File") + " (*.geom;*.bgeom);;" + tr("All Files (*.*)"),
                 [this](const QString& filename) {
    if (saveAsGeom(filename)) setFilename(filename);
  });
}

void
ViewGeomSceneGL::saveAsGeom()
{
  selectGeomFilenames([this](const QString& shape, const QString& geom, const QString& mat) {
    if (saveScene(shape, geom, mat, __scene)) setFilename(shape);
  });
}

void
ViewGeomSceneGL::saveSelection()
{
  if(__selectedShapes.empty()){
    QMessageBox::warning(__frame,tr("Empty Selection"),tr("No Shape are selected!"),
                                   "Ok");
    return;
  }
  selectGeomFilenames([this](const QString& shape, const QString& geom, const QString& mat) {
    saveScene(shape, geom, mat, getSelection());
  });
}

void
ViewGeomSceneGL::saveNotSelection()
{
  if(__selectedShapes.empty()){
    QMessageBox::warning(__frame,tr("Empty Selection"),tr("No Shape are selected!"),
                                   "Ok");
    return;
  }
  selectGeomFilenames([this](const QString& shape, const QString& geom, const QString& mat) {
    saveScene(shape, geom, mat, getNotSelection());
  });
}

/* ----------------------------------------------------------------------- */
bool
ViewGeomSceneGL::saveAsGeom(const QString& filename)
{
  return saveScene(filename,QString(),QString(),__scene);
}

bool
ViewGeomSceneGL::saveAsGeom(const QString& shape,
                            const QString& geom,
                            const QString& mat)
{
  return saveScene(shape,geom,mat,__scene);
}

/*
bool
ViewGeomSceneGL::saveAs(const QString& filename){
  QString extension=filename.right(filename.length()-filename.findRev('.')-1);
  extension= extension.upper();
  if(extension == "GEOM" ){
    QString _geomFilename = filename;
    QString _matFilename = filename.left(filename.findRev('.')+1) + "mat";
    ofstream fgeom(_geomFilename);
    fgeom << " # -Geom File-\n" ;
    ofstream fmat(_matFilename);
    fmat << " # -Appearance File-\n";
    Printer _printer(fgeom,fgeom,fmat);
    _printer.header(fgeom,_geomFilename,"File Generated with Geom 3D Viewer");
    fgeom  << ":include " << _matFilename.1240().constData() << "\n\n\n";
    _printer.header(fmat,_matFilename,"File Generated with Geom 3D Viewer");
    return __scene->apply(_printer);
  }
  else if(extension == "BGEOM" ){
    return BinaryPrinter::print(__scene,string(filename.data()),"File Generated with Geom 3D Viewer");
  }
  else return false;
}*/

class GeomDialog : public QFileDialog {
public :
  GeomDialog( const QString & initial,
              const QString & filter = QString(),
              QWidget * parent=0,
              const char * name = 0,
              bool modal = true ) :
  QFileDialog ( parent, name, !initial.isEmpty()?QFileInfo(initial).filePath():"", filter   ) {
    // setSelection(initial);
    setFileMode(QFileDialog::AnyFile);
    QGroupBox * box = new QGroupBox(qApp->translate("ViewGeomSceneGL","Separated File for"),this);
    __geometry = new QRadioButton(qApp->translate("ViewGeomSceneGL","Geometry"),box);
    __appearance = new QRadioButton(qApp->translate("ViewGeomSceneGL","Appearance"),box);
    //setextension(box);
  }

  bool geometrySeparated() const {return __geometry->isChecked();}
  bool appearanceSeparated() const {return __appearance->isChecked();}

protected:

  QRadioButton *  __geometry;
  QRadioButton *  __appearance;
};

void
ViewGeomSceneGL::selectGeomFilenames(
  const std::function<void(const QString&, const QString&, const QString&)>& onSelected)
{
  QString initial = getFilename();
  QString extension=getFilename().right(getFilename().length()-getFilename().lastIndexOf('.')-1);
  extension= extension.toUpper();
  if(!extension.isEmpty() && extension != "GEOM" && extension != "BGEOM" ){
    initial = getFilename().left(getFilename().lastIndexOf('.')+1);
    initial += "geom";
  }

  auto * dialog = new GeomDialog(initial, tr("PlantGL File") + " (*.geom;*.bgeom);;" +
                                 tr("All Files (*.*)"), __frame->window(), qPrintable(tr("Save")));
  dialog->setAcceptMode(QFileDialog::AcceptSave);
  dialog->setOption(QFileDialog::DontConfirmOverwrite);
  dialog->setDefaultSuffix("geom");
  connect(dialog, &QFileDialog::fileSelected, this, [this, dialog, onSelected](const QString& filename) {
    const bool separateGeometry = dialog->geometrySeparated();
    const bool separateAppearance = dialog->appearanceSeparated();
    QTimer::singleShot(0, this, [this, onSelected, filename, separateGeometry, separateAppearance]() {
      const QFileInfo file(filename);
      const bool binary = file.suffix().compare("bgeom", Qt::CaseInsensitive) == 0;
      const QString base = file.dir().filePath(file.completeBaseName());
      const QString shape = binary || file.suffix().compare("geom", Qt::CaseInsensitive) == 0
                          ? filename : base + ".geom";
      const QString geom = !binary && separateGeometry ? base + "_geom.geom" : QString();
      const QString mat = !binary && separateAppearance ? base + ".mat" : QString();

      QStringList existingFiles;
      for (const QString& path : {shape, geom, mat}) {
        if (QFile::exists(path)) existingFiles.append(path);
      }
      for (const QString& path : existingFiles) {
        QMessageBox::StandardButtons buttons = QMessageBox::Yes | QMessageBox::Cancel;
        if (existingFiles.size() > 1) buttons |= QMessageBox::YesToAll;
        const auto answer = QMessageBox::warning(__frame->window(), tr("File Already Exists"),
            tr("%1 already exists. Overwrite?").arg(path), buttons, QMessageBox::Cancel);
        if (answer == QMessageBox::YesToAll) break;
        if (answer != QMessageBox::Yes) return;
      }
      onSelected(shape, geom, mat);
    });
  });
  showFileDialog(dialog);
}

bool
ViewGeomSceneGL::saveScene(const QString& shape,
                           const QString& geom,
                           const QString& mat,
                           PGL(ScenePtr) scene){
  if(shape.isEmpty())return false;
  QString extension=shape.right(shape.length()-shape.lastIndexOf('.')-1);
  extension= extension.toUpper();
  if(extension == "GEOM" ){
      ofstream fshape(qPrintable(shape));
    if(!fshape)return false;
    fshape << " # -Geom File-\n";
    ofstream * fgeom;
    bool separatedgeom = false;
    if(!geom.isEmpty()){
      fgeom = new ofstream(qPrintable(geom));
      if(!*fgeom)fgeom = &fshape;
      else {
        *fgeom << " # -Geom File-\n";
        separatedgeom = true;
      }
    }
    else fgeom = &fshape;

    ofstream * fmat;
    bool separatedmat = false;
    if(!mat.isEmpty()){
      fmat = new ofstream(qPrintable(mat));
      if(!*fmat)fmat = &fshape;
      else {
        *fmat << " # -Appearance File-\n";
        separatedmat = true;
      }
    }
    else fmat = &fshape;

    Printer _printer(fshape,*fgeom,*fmat);
    _printer.header(fshape,qPrintable(shape),qPrintable(tr("File Generated with PlantGL 3D Viewer")));
    if(separatedgeom){
      _printer.header(*fgeom,qPrintable(geom),qPrintable(tr("File Generated with PlantGL 3D Viewer")));
      fshape  << ":include " << geom.toStdString() << "\n\n\n";
    }
    if(separatedmat){
      _printer.header(*fmat,qPrintable(mat),qPrintable(tr("File Generated with PlantGL 3D Viewer")));
      fshape  << ":include " << mat.toStdString() << "\n\n\n";
    }
    return scene->apply(_printer);
  }
  else if(extension == "BGEOM" ){
      return BinaryPrinter::print(__scene,qPrintable(shape),qPrintable(tr("File Generated with PlantGL 3D Viewer")));
  }
  else {
      if (!__scene->save(shape.toStdString())){
        QMessageBox::warning(__frame,tr("Cannot save"),QString(tr("Problem with saving file.")),1,0,0);
        return false;
      }
      return true;
  }
}


/* ----------------------------------------------------------------------- */
void
ViewGeomSceneGL::saveAsPovRay()
{
  selectSaveFile(__frame, this, getFilename(), "pov", tr("PovRay File")+" (*.pov);;"+tr("All Files (*.*)"),
                 [this](const QString& filename) {
    saveAsPovRay(filename);
  });
}

void
ViewGeomSceneGL::saveAsVegeStar()
{
  selectSaveFile(__frame, this, getFilename(), "vgx", tr("VegeStar File")+" (*.vgx;*.vgs);;"+tr("All Files (*.*)"),
                 [this](const QString& filename) {
    saveAsVegeStar(filename);
  });
}

void
ViewGeomSceneGL::saveAsAmapSymbol()
{
  selectSaveFile(__frame, this, getFilename(), "smb", tr("Amap Symbol")+" (*.smb);;"+tr("All Files (*.*)"),
                 [this](const QString& filename) {
    saveAsAmapSymbol(filename);
  });
}

void
ViewGeomSceneGL::saveAsLinetree(){
  auto * dialog = new ViewReadLinetree(false, __frame->window());
  const QString filename = getFilename();
  if (!filename.isEmpty()) {
    const QFileInfo file(filename);
    dialog->setLigFile(file.suffix().compare("lig", Qt::CaseInsensitive) == 0
                       ? filename : file.dir().filePath(file.completeBaseName() + ".lig"));
  }
  connect(dialog, &QDialog::accepted, this, [this, dialog]() {
    const QString lig = dialog->getLigFile();
    const QString dta = dialog->getDtaFile();
    const QString smb = dialog->getSMBPath();
    const QString key = dialog->getKey();
    const bool bigEndian = dialog->bigEndian();
    QTimer::singleShot(0, this, [this, lig, dta, smb, key, bigEndian]() {
      saveAsLinetree(lig, dta, smb, key, bigEndian);
    });
  });
  showFileDialog(dialog);
}

bool
ViewGeomSceneGL::saveAsLinetree(const QString& filename){
  ViewReadLinetree __linetreeDialog(false);
  if(!filename.isEmpty())__linetreeDialog.setLigFile(filename);
  if(__linetreeDialog.exec()){
    return saveAsLinetree(__linetreeDialog.getLigFile(),
                          __linetreeDialog.getDtaFile(),
                          __linetreeDialog.getSMBPath(),
                          __linetreeDialog.getKey(),
                          __linetreeDialog.bigEndian());
  }
  else return false;
}

/* ----------------------------------------------------------------------- */

bool
ViewGeomSceneGL::saveAsPovRay(const QString& filename)
{
  ofstream _stream(qPrintable(filename));
  if(_stream){
    Tesselator _tess;
    PovPrinter _printer(_stream,_tess);
    _printer.header(qPrintable(tr("File Generated with PlantGL 3D Viewer")));
    _printer.beginHeader();
    _printer.setCamera(__camera->getEye(),__camera->getTranslation(),__camera->getViewAngle(),__camera->getAzimuth(),__camera->getElevation());
    const QColor& col = __light->getDiffuse();
    _printer.setLight(__light->getPosition(),Color3(col.red(),
                            col.green(),
                            col.blue()));
    _stream << "/*\n";
    _printer.setLight(__camera->getPosition(),Color3(col.red(),
                                                    col.green(),
                                                    col.blue()));
    _stream << "*/\n\n";
    const QOpenGLBaseContext * c =__frame->context();
    if(c){
      // QColor bg = c->overlayTransparentColor();
      // if(bg.isValid())_printer.setBackGround(Color3(bg.red(),bg.green(),bg.blue()));
      //else {
        ViewGLFrame * f = dynamic_cast<ViewGLFrame *>(__frame);
        if(f != NULL){
            QColor bg = f->getBackGroundColor();
            _printer.setBackGround(Color3(bg.red(),bg.green(),bg.blue()));
            _printer.setLineWidth(f->getLineWidth()/10);
        }
      //}
    }
    _printer.endHeader();
    _stream << "\n\n";
    __scene->apply(_printer);
    return true;
  }
  else return false;
}


bool
ViewGeomSceneGL::saveAsVegeStar(const QString& filename)
{
  ofstream _stream(qPrintable(filename));
  if(_stream){
    Tesselator _tess;
    VgstarPrinter _printer(_stream,_tess);
    _printer.header(qPrintable(tr("File Generated with PlantGL 3D Viewer")));
    __scene->apply(_printer);
    setFilename(filename);
    return true;
  }
  else return false;
}


bool
ViewGeomSceneGL::saveAsAmapSymbol(const QString& filename)
{
  Shape3DPtr shape = __scene->getAt(0);
  AmapTranslator translator(__discretizer);
  shape->apply(translator);
  AmapSymbolPtr result = translator.getSymbol();

  if(result) {
      beofstream _stream(qPrintable(filename));
      if(_stream){
          result->write(_stream);
          setFilename(filename);
          return true;
      }
      else return false;
  }
  else return false;
}


bool
ViewGeomSceneGL::saveAsLinetree(const QString& lig,
                                const QString& dta,
                                const QString& smb,
                                const QString& key,
                                bool bigendian){
    bofstream * ligstream;
    if (bigendian) ligstream = new beofstream(qPrintable(lig));
    else ligstream = new leofstream(qPrintable(lig));
    if(!*ligstream) return false;
    ofstream dtastream(qPrintable(dta));
    if(!dtastream) return false;
    AmapTranslator t(__discretizer);
    LinetreePrinter printer(*ligstream,dtastream,qPrintable(smb),t);
    printer.header(qPrintable(key));
    if(__scene->apply(printer)){
        setFilename(lig);
        QString inf = lig.left(lig.lastIndexOf('.')+1);
        inf += "inf";
        ofstream infstream(qPrintable(inf));
        if(infstream){
            printer.printInf(infstream,
                             __bbox->getUpperRightCorner(),
                             __bbox->getLowerLeftCorner());
        }
        QString cfg = QFileInfo(lig).dir().path();
        string _cfg;
        if(!printer.printCfg(qPrintable(cfg),".",qPrintable(smb),_cfg)){
            QMessageBox::warning(__frame,tr("Cfg File"),
            tr("Cannot write file")+" '"+QString(_cfg.c_str())+"'",
            "Ok");
        }
        delete ligstream;
        return true;
    }
    else {
        delete ligstream;
        return false;
    }
}



void
ViewGeomSceneGL::saveAsVrml()
{
  selectSaveFile(__frame, this, getFilename(), "wrl", tr("Vrml File")+" (*.wrl);;"+tr("All Files (*.*)"),
                 [this](const QString& filename) {
    ofstream _stream(qPrintable(filename));
    if(_stream){
      VrmlPrinter _printer(_stream,__discretizer);
      _printer.header(qPrintable(tr("File Generated with PlantGL 3D Viewer")));
      if(fabs(__camera->getAzimuth()) > GEOM_EPSILON   ||
         fabs(__camera->getElevation()) > GEOM_EPSILON ||
         __camera->getTranslation() != Vector3::ORIGIN)
      _printer.setCamera(__camera->getEye()-__camera->getTranslation(),__camera->getAzimuth(),__camera->getElevation(),"Geom View");
      _printer.setCamera(__camera->getEye(),0,0,"Home");
      const QColor& col = __light->getDiffuse();
      const QColor& col2 = __light->getAmbient();
      _printer.setLight(__light->getPosition(),Color3(col2.red(),col2.green(),col2.blue()),
                        Color3(col.red(),col.green(),col.blue()),4*norm(__light->getPosition()));
      const QOpenGLBaseContext * c =__frame->context();
      //if(c){
        //QColor bg = c->overlayTransparentColor();
        //if(bg.isValid())_printer.setBackGround(Color3(bg.red(),bg.green(),bg.blue()));
        //else {
        //  ViewGLFrame * f = dynamic_cast<ViewGLFrame *>(__frame);
        //  if(f != NULL){
        //      bg = f->getBackGroundColor();
        //      _printer.setBackGround(Color3(bg.red(),bg.green(),bg.blue()));
        //  }
      //  }
      //}
      __scene->apply(_printer);
    }
  });
}

void
ViewGeomSceneGL::saveAsPly()
{
  selectSaveFile(__frame, this, getFilename(), "ply", tr("Ply File")+ " (*.ply);;"+tr("All Files (*.*)"),
                 [this](const QString& filename) {
    PlyPrinter::ply_format _format = PlyPrinter::ply_ascii;
    switch(QMessageBox::warning(__frame,"Ply Format",tr("Ply File Format"),
                                tr("Ascii"),tr("Binary Little Endian"),tr("Binary Big Endian"))){
    case 0 :
      _format = PlyPrinter::ply_ascii;
      break;
    case 1 :
      _format = PlyPrinter::ply_binary_little_endian;
      break;
    case 2 :
      _format = PlyPrinter::ply_binary_big_endian;
      break;
    case -1 :
      return;
      break;
    }
    Tesselator t;
    PlyPrinter::print(__scene,t,filename.toStdString(),
      qPrintable(tr("File Generated with PlantGL 3D Viewer")),_format);
  });
}



/* ----------------------------------------------------------------------- */


bool
ViewMultiGeomSceneGL::openGeomFiles(const QString& filename,const QString& filename2)
{
  if(!filename.isEmpty() && !filename2.isEmpty()){
#ifdef GEOM_THREAD
    if(__reader && __reader->isRunning()){
      QMessageBox::warning(__frame,tr("Already Reading File"),
                           QString(tr("Currently Reading File")+" %1").
                           arg(__reader->getFilename()),1,0,0);
      return false;
    }
    else {
      if(__reader) {
        delete __reader;
        __reader = NULL;
      }
      __reader = new ViewMultiGeomReader(filename,
                                         filename2,
                                         this);
      __reader->start();
      return true;
    }
#else
    stringstream _errlog(ios::out) ;
    ScenePtr scene1 = ScenePtr(new Scene(qPrintable(filename),"",_errlog));
    ScenePtr scene2 = ScenePtr(new Scene(qPrintable(filename2),"",_errlog));
    setScene(scene1,scene2);
    string _msg = _errlog.str();
    if(!_msg.empty())error(_msg.c_str());
    emit valueChanged();
    setFilename(filename);
    if(__transSlider)__transSlider->show();
    return true;
#endif
  }
  return false;
}

void
ViewMultiGeomSceneGL::openGeomFiles()
{
  const QString caption = tr("Open PlantGL");
  const QString filter = tr("PlantGL File") + " (*.geom;*.bgeom);;" + tr("All Files (*.*)");
  selectExistingFile(__frame, this, caption, getFilename(), filter,
                     [this, caption, filter](const QString& firstFilename) {
    selectExistingFile(__frame, this, caption, getFilename(), filter,
                       [this, firstFilename](const QString& secondFilename) {
      openGeomFiles(firstFilename, secondFilename);
    });
  });
}


bool
ViewMultiGeomSceneGL::addOpenEntries(QMenu * menu)
{
  QPixmap openIcon( ViewerIcon::getPixmap(ViewerIcon::fileopen) );
  menu->addAction( openIcon, tr("Open &Geom File"),  this,SLOT(openGeomFile()),Qt::CTRL | Qt::Key_G);
  menu->addAction( openIcon, tr("&Add Geom File"),   this,SLOT(addGeomFile()));
  menu->addAction( openIcon, tr("Open &2 Geom File"),this,SLOT(openGeomFiles()));
  return true;
}

/* ----------------------------------------------------------------------- */
