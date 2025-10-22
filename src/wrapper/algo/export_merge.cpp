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



#include <boost/python.hpp>

#include <plantgl/algo/base/merge.h>
#include <plantgl/algo/base/discretizer.h>

#include <plantgl/python/export_property.h>

PGL_USING_NAMESPACE
using namespace boost::python;
using namespace std;

ExplicitModelPtr merge_geometry(GeometryPtr a, GeometryPtr b){
  Discretizer d;
  a->apply(d);
  Merge m(d,d.getDiscretization());
  m.apply(b);
  return m.getModel();
}

void export_Merge()
{
  class_< Merge, boost::noncopyable >
    ("Merge", init<Discretizer&,ExplicitModelPtr&>("Merge(Discretizer d, ExplicitModel e )" ))
    .DEC_PTR_PROPERTY_RO(model,Merge,Model,ExplicitModelPtr)
    .def("apply", ( bool(Merge::*)(GeometryPtr&)      ) &Merge::apply)
    .def("apply", ( bool(Merge::*)(ExplicitModelPtr&) ) &Merge::apply)
    .add_property("result",make_function(&Merge::getModel,return_internal_reference<1>()))
    ;
  def("merge_geometry",&merge_geometry);
}


