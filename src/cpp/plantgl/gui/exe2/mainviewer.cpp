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

#include <QApplication>
#include <QLabel>
#include <QSurfaceFormat>

#ifndef QT_NO_OPENGL
#include "../viewer2/mainwidget.h"
#include <plantgl/pgl_scenegraph.h>
#endif

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setVersion(4, 5);
    format.setDepthBufferSize(24);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setOption(QSurfaceFormat::DebugContext);

    QSurfaceFormat::setDefaultFormat(format);

    app.setApplicationName("pglviewer2");
    app.setApplicationVersion("0.1");

#ifndef QT_NO_OPENGL
    MainWidget widget;
    ScenePtr scene(new Scene());
    if (argc > 1) {
        printf("***** %s\n", argv[1]);
        scene = new Scene(argv[1]);
    }
    else {
        printf("***** default\n");
        // scene->add(ShapePtr(new Shape(GeometryPtr(new Sphere(3, 128, 128)), AppearancePtr(new Material(Color3(50,10,10), 5)) )));
        scene->add(ShapePtr(new Shape(GeometryPtr(new Cylinder(0.5, 3, false, 128)), AppearancePtr(new Material(Color3(50,10,10), 5)) )));
    }
    widget.setScene(scene);
    widget.show();
#else
    QLabel note("OpenGL Support required");
    note.show();
#endif
    return app.exec();
}





/*
#include "../viewer/pglapplication.h"

int main( int argc, char **argv )
{

    PGLViewerApplication::init();
    PGLViewerApplication::start();
    // PGLViewerApplication::wait(10000);
    PGLViewerApplication::exit();

  return 0;
}

*/
