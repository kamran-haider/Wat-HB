/*
  C helper module for intensive calcs.
#=============================================================================================
# VERSION CONTROL INFORMATION
#=============================================================================================
__version__ = "$Revision: $ $Date: $"
# $Date: $
# $Revision: $
# $LastChangedBy: $
# $HeadURL: $
# $Id: $

#=============================================================================================

#=============================================================================================
# INSTALLATION INSTRUCTIONS
#=============================================================================================

  To compile on Linux:

  gcc -O3 -lm -fPIC -shared -I(directory with Python.h) -I(directory with numpy/arrayobject.h) -o _gistcalcs.so _gistcalcs.c

  For a desmond installation of python 2.5 (change path up to desmond directory, rest should be the same):
  
  gcc -O3 -lm -fPIC -shared -I /home/kamran/desmond/mmshare-v24012/lib/Linux-x86_64/include/python2.7 -I /home/kamran/desmond/mmshare-v24012/lib/Linux-x86_64/lib/python2.7/site-packages/numpy/core/include/ -o _gistcalcs.so _gistcalcs.c
  
  For a default installation of python 2.5:
  
  gcc -O3 -lm -fPIC -shared -I/usr/local/include/python2.5 -I/usr/local/lib/python2.5/site-packages/numpy/core/include -o _gistcalcs.so _gistcalcs.c

/Users/Kamran/anaconda/include/python2.7/Python.h
/Users/Kamran/anaconda/pkgs/numpy-1.9.0-py27_0/lib/python2.7/site-packages/numpy/core/include/
*/
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double dist_mic(double x1, double x2, double x3, double y1, double y2, double y3, double b1, double b2, double b3) {
    /* Method for obtaining inter atom distance using minimum image convention
     */
    double dx, dy, dz;
    dx = x1-y1;
    dy = x2-y2;
    dz = x3-y3;
    if (dx > b1/2.0) dx -= b1; 
    else if (dx < -b1/2.0) dx += b1; 
    if (dy > b2/2.0) dy -= b2;
    else if (dy < -b2/2.0) dy += b2;
    if (dz > b3/2.0) dz -= b3; 
    else if (dz < -b3/2.0) dz += b3;

    return sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
    }
    
double dist(double x1, double x2, double x3, double y1, double y2, double y3) {
    /* Method for Euclidean distance between two points
     */
    double dx, dy, dz;
    dx = x1-y1;
    dy = x2-y2;
    dz = x3-y3;
    return sqrt(pow(dx, 2)+ pow(dy, 2)+ pow(dz, 2));
    }



PyObject *_hbcalcs_processGrid(PyObject *self, PyObject *args)
    {
    // variables reterived from python
    int frames, start_frame, num_atoms; // total number of frames and atoms in the system
    PyObject *getCoords, *arglist_getCoords; // python function, its argument tuple and variable to store its results
    PyObject *sendWatCoords, *arglist_sendWatCoords;// python function, its argument tuple and variable to store its results
    // Following variables are pointers to arrays that come from Python and contain various important pieces of info
    PyObject *coords; // coordinates for all atoms, retrieved through a python callback, are stored in this array
    PyArrayObject *all_at_ids, *solute_at_ids, *wat_oxygen_ids, *wat_all_ids;
    PyArrayObject *charges, *vdw, *box, *grid_dim, *grid_orig;
    PyArrayObject *voxel_data; // array for voxel, arrives here empty
    PyArrayObject *wat_index_info;
    PyArrayObject *test_array;

    // variables to be used locally
    int i_frames, i_wat; // frame counter, water counter
    int n_wat;    
    double grid_max_x, grid_max_y, grid_max_z;
    double grid_orig_x, grid_orig_y, grid_orig_z;
    double grid_index_x, grid_index_y, grid_index_z; 
    double *wat_x, *wat_y, *wat_z;
    double *h1x, *h1y, *h1z, *h2x, *h2y, *h2z;
    int *wat_id;
    double wat_dist;
    int n_atomic_sites, n_pseudo_sites, wat_begin_id, pseudo_begin_id, oxygen_index;

    // Argument parsing to reterive everything sent from Python correctly    
    if (!PyArg_ParseTuple(args, "iiiOO!O!O!O!O!O!O!O!O!O!O!:processGrid",
                            &frames, &start_frame, &num_atoms, &getCoords,
                            &PyArray_Type, &wat_index_info,
                            &PyArray_Type, &all_at_ids,
                            &PyArray_Type, &solute_at_ids,
                            &PyArray_Type, &wat_oxygen_ids,
                            &PyArray_Type, &wat_all_ids,
                            &PyArray_Type, &charges,
                            &PyArray_Type, &vdw,
                            &PyArray_Type, &box,
                            &PyArray_Type, &grid_dim,
                            &PyArray_Type, &grid_orig,
                            &PyArray_Type, &voxel_data))
        {
            return NULL; /* raise argument parsing exception*/
        }
    // Consistency checks for Python Callback
    if (!PyCallable_Check(getCoords)) {
        PyErr_Format(PyExc_TypeError,
                     "function is not callcable");
        return NULL;
        }

    // Set grid max points and grid origin
    grid_max_x = *(double *)PyArray_GETPTR1(grid_dim, 0) * 0.5 + 1.5;
    grid_max_y = *(double *)PyArray_GETPTR1(grid_dim, 1) * 0.5 + 1.5;
    grid_max_z = *(double *)PyArray_GETPTR1(grid_dim, 2) * 0.5 + 1.5;
    grid_orig_x = *(double *)PyArray_GETPTR1(grid_orig, 0);
    grid_orig_y = *(double *)PyArray_GETPTR1(grid_orig, 1);
    grid_orig_z = *(double *)PyArray_GETPTR1(grid_orig, 2);
    printf("grid origin: %f %f %f \n", grid_orig_x , grid_orig_y, grid_orig_z);
    printf("grid max: %f %f %f \n", grid_max_x , grid_max_y, grid_max_z);
    printf("grid dim: %i %i %i \n", (int)*(double *)PyArray_GETPTR1(grid_dim, 0), (int)*(double *)PyArray_GETPTR1(grid_dim, 1), (int)*(double *)PyArray_GETPTR1(grid_dim, 2));

    // Parse index information array
    n_atomic_sites = *(int *)PyArray_GETPTR1(wat_index_info, 0);
    n_pseudo_sites = *(int *)PyArray_GETPTR1(wat_index_info, 1);
    wat_begin_id = *(int *)PyArray_GETPTR1(wat_index_info, 2);
    pseudo_begin_id = *(int *)PyArray_GETPTR1(wat_index_info, 3);
    oxygen_index = *(int *)PyArray_GETPTR1(wat_index_info, 4);
    n_wat = PyArray_DIM(wat_oxygen_ids, 0);
    //printf("nwat: %i \n", n_wat);
    
    for (i_frames = start_frame; i_frames < frames + start_frame; i_frames ++) {
        printf("processing frame: %i\n", i_frames+1);
        // voxel_id initialized to zero (for each water it's voxel ID will be stored in this variable) 
        int voxel_id = 0;
        // for each frame get coordinates of every atom from python
        arglist_getCoords = Py_BuildValue("(i)", i_frames);
        coords = PyEval_CallObject(getCoords, arglist_getCoords);

        // Now we need to iterate over every water atom inside grid for its voxel assignment
        for (i_wat = 0; i_wat < n_wat; i_wat ++) {
            wat_id = (int *) PyArray_GETPTR1(wat_oxygen_ids, i_wat); // obtain index for this atom (this is not array index, this is unique atom id)
            // use water ID to get the correct x, y, z coordinates from coord array
            wat_x = (double *)PyArray_GETPTR2(coords, *wat_id-1, 0);
            wat_y = (double *)PyArray_GETPTR2(coords, *wat_id-1, 1); 
            wat_z = (double *)PyArray_GETPTR2(coords, *wat_id-1, 2);
            //printf("water oxygen ID %i and coordinates %f %f %f\n", *wat_id, *wat_x, *wat_y, *wat_z);
            // check if the distance between wateer coordinates and grid origin is less than the max grid point
            // this means do calculations only waters inside the grid
            if (*wat_x - grid_orig_x <= grid_max_x && *wat_y - grid_orig_y <= grid_max_y && *wat_z - grid_orig_z <= grid_max_z &&
                *wat_x - grid_orig_x >= -1.5 && *wat_y - grid_orig_y >= -1.5 && *wat_z - grid_orig_z >= -1.5){
                    //printf("water %i is inside the grid!\n", *wat_id);
                if (*wat_x - grid_orig_x >= 0 && *wat_y - grid_orig_y >= 0 && *wat_z - grid_orig_z >= 0){
                    // transform water coordinates in units of grid dimensions
                    grid_index_x = (*wat_x - grid_orig_x)/0.5;
                    grid_index_y = (*wat_y - grid_orig_y)/0.5;
                    grid_index_z = (*wat_z - grid_orig_z)/0.5;
                    // check if water coords (in grid dimensions) are less than grid dimensions in each direction
                    if (grid_index_x < (int)*(double *)PyArray_GETPTR1(grid_dim, 0) &&
                        grid_index_y < (int)*(double *)PyArray_GETPTR1(grid_dim, 1) &&
                        grid_index_z < (int)*(double *)PyArray_GETPTR1(grid_dim, 2)){
                        // obtain the voxel ID for this water
                        voxel_id = ((int)grid_index_x*(int)*(double *)PyArray_GETPTR1(grid_dim, 1) + (int)grid_index_y)*(int)*(double *)PyArray_GETPTR1(grid_dim, 2) + (int)grid_index_z;
                        // Energy calculations
                        //energy(wat_x, wat_y, wat_z, solute_at_ids, wat_oxygen_ids, coords, charges, vdw, box, voxel_data, wat_index_info, *wat_id, voxel_id);
                        //energy_ww(wat_x, wat_y, wat_z, wat_oxygen_ids, coords, charges, vdw, box, voxel_data, *wat_id, voxel_id, n_wat);
                        // get hydrogen atom coords
                        h1x = (double *)PyArray_GETPTR2(coords, *wat_id, 0);
                        h1y = (double *)PyArray_GETPTR2(coords, *wat_id, 1); 
                        h1z = (double *)PyArray_GETPTR2(coords, *wat_id, 2);
                        h2x = (double *)PyArray_GETPTR2(coords, *wat_id + 1, 0);
                        h2y = (double *)PyArray_GETPTR2(coords, *wat_id + 1, 1); 
                        h2z = (double *)PyArray_GETPTR2(coords, *wat_id + 1, 2);
                        //printf("Energy value for this voxel: %f\n",*(double *)PyArray_GETPTR2(voxel_data, voxel_id, 13));
                        //printf("water coords %f %f %f\n", *wat_x, *wat_y, *wat_z);
                        // send water x, y, z to python
                        //arglist_sendWatCoords = Py_BuildValue("(iddddddddd)", voxel_id, *wat_x, *wat_y, *wat_z, *h1x, *h1y, *h1z, *h2x, *h2y, *h2z);
                        //PyEval_CallObject(sendWatCoords, arglist_sendWatCoords);
                        //printf("water coords %f %f %f\n", *wat_x, *wat_y, *wat_z);
                        //printf("grid indices %f %f %f\n", grid_index_x, grid_index_y, grid_index_z);
                        //printf("grid indices %i %i %i\n", (int)grid_index_x, (int)grid_index_y, (int)grid_index_z);
                        //printf("(%i*%i + %i)*%i + %i = ", (int)grid_index_x, (int)*(double *)PyArray_GETPTR1(grid_dim, 1), (int)grid_index_y, (int)*(double *)PyArray_GETPTR1(grid_dim, 2), (int)grid_index_z);
                        //printf("voxel id: %i\n", voxel_id);
                        // Once voxel id is obtained, it is used to reterieve various properties of the voxel and modify them based
                        // based on the calculations. Here, we reterive voxel water population and raise it by 1
                        *(double *)PyArray_GETPTR2(voxel_data, voxel_id, 4) += 1.0;
                        // insert code here that assigns hydrogens correctly
                        *(double *)PyArray_GETPTR2(voxel_data, voxel_id, 6) += 2.0;
                        //*(int *) PyArray_GETPTR2(voxel_data, voxel_id, 3) += 1.0;
                        // 
                        //printf("voxel id: %i\n", voxel_id);
                        
                        }
                    }
                }
            //printf("wat_id x y z max_x max_y max_z: %i %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f \n", *wat_id, *wat_x, *wat_y, *wat_z, *grid_max_x, *grid_max_y, *grid_max_z);
            }
        //printf("waters inside grid! %i\n", frame_wat);
        }
    return Py_BuildValue("i", 1);
    
    }



/* Method Table
 * Registering all the functions that will be called from Python
 */

static PyMethodDef _hbcalcs_methods[] = {
    {
        "processGrid",
        (PyCFunction)_hbcalcs_processGrid,
        METH_VARARGS,
        "Process grid"
    },
    {NULL, NULL}
};

/* Initialization function for this module
 */

PyMODINIT_FUNC init_hbcalcs() // init function has the same name as module, except with init prefix
{
    // we produce name of the module, method table and a doc string
    Py_InitModule3("_hbcalcs", _hbcalcs_methods, "Process GIST calcs.\n");
    import_array(); // required for Numpy initialization
}
