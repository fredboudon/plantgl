# -*- coding: iso-8859-15 -*-


import os, sys
from setuptools import setup
pj = os.path.join


# Setup script

name = 'plantgl'
namespace = 'openalea'
pkg_name = 'openalea.plantgl'

version= '2.0.1'

description= 'PlantGL package for OpenAlea.' 
long_description= '''
Plant Geometric Library is a powerfull library to create and display vegetal 3D scene.
'''

author= 'Frederic Boudon, Christophe Pradal'
author_email= 'frederic.boudon@cirad.fr, christophe.pradal@cirad.fr'
url= 'http://openalea.gforge.inria.fr/dokuwiki/doku.php?id=packages:visualization:plantgl:plantgl'

license= 'Cecill-C' 

# Scons build directory
build_prefix= "build-scons"


# Main setup
setup(
    name="PlantGL",
    version=version,
    description=description,
    long_description=long_description,
    author=author,
    author_email=author_email,
    url=url,
    license=license,
    
    # Define what to execute with scons
    # scons is responsible to put compiled library in the write place
    # ( lib/, package/, etc...)
    scons_scripts = ['SConstruct'],
    scons_parameters = ["build_prefix="+build_prefix],

    namespace_packages = ["openalea"],
    create_namespaces = True,

    # pure python  packages
    packages= [ pkg_name, pkg_name+'.math', pkg_name+'.scenegraph', pkg_name+'.algo',
                pkg_name+'.gui', pkg_name+'.gui3', pkg_name+'.wralea', pkg_name+'.ext',
                pkg_name+'.codec'],
    
    # python packages directory
    package_dir= { pkg_name : pj('src',name),
                   pkg_name+'.math' :pj( 'src', name, 'math' ),
                   pkg_name+'.scenegraph' :pj( 'src', name, 'scenegraph' ),
                   pkg_name+'.algo' :pj( 'src', name, 'algo' ),
                   pkg_name+'.gui' :pj( 'src', name, 'gui' ),
                   pkg_name+'.gui3' :pj( 'src', name, 'gui3' ),
                   pkg_name+'.wralea' :pj( 'src', name, 'wralea' ),
                   pkg_name+'.ext' :pj( 'src', name, 'ext' ),
                   pkg_name+'.codec' :pj( 'src', name, 'codec' ),
                   },

                   
    # Add package platform libraries if any
    include_package_data = True,
    package_data = {'' : ['*.pyd'],},
    zip_safe = False,

    # Specific options of openalea.deploy
    lib_dirs = {'lib' : pj(build_prefix, 'lib'),},
    bin_dirs = {'bin':  pj(build_prefix, 'bin'),},
    inc_dirs = { 'include' : pj(build_prefix, 'include') },
    postinstall_scripts = ['openalea.plantgl.postinstall',],


    # Scripts
    #entry_points = { 'gui_scripts': [ 'pglviewer = openalea.plantgl:start_viewer',]},
 
    # Dependencies
    setup_requires = ['openalea.deploy'],
    dependency_links = ['http://openalea.gforge.inria.fr/pi'],
    install_requires = [],


    )



    
