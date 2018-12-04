/*
 * Lattice Boltzmann
 * On-grid Bounce back: Wall ; Cylinder body
 * Zou-He Velocity & Pressure: Inlet & Outlet
 * 
 */

/* 
 * File:   main.cpp
 * Author: jiayin
 *
 * Created on November 30, 2018, 12:32 PM
 */

#include <iostream>
#include <cmath>
#include <string>
#include "omp.h"
#include <fstream>

using namespace std;


/*
 * Customizable set up: 
 * H, L: Channel Height, Length,;
 * r: radius of cylinder;
 * tau: relaxation time parameter
 * Re: Reynold's number
 * nt: number of threads to run in parallel computing in OpenMP
 */
const int H = 20;
const int L = 50;
const double r = 4.0;
const double tau = 0.6;
const double Re = 10.0;
const int nt = 25;  

/*pg: Poiseuille Flow Channel Grid*/
/*mg: Mesh Grid setup with cylinder immersed*/
/*dx = 1; dt = 1; c = dx/dt = 1*/
const int n = H+3;
const int m = L+1;
int pg[n][m];
int mg[n][m];
double c = 1.0;
double vis = (2.*tau-1)/6.0;
double delta_p = -36.0 * Re* pow(vis,2.) *L /pow(H,3.);
double p0 = 1. - delta_p/2.0;
double p1 = 1. + delta_p/2.0;

/*Generate mg and pg*/
void createGrid(){
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < m; k++){
            
            if (pow(j-1.0/2.*(H+2.), 2.) + pow((k-1.0/4.*L), 2.) < pow(r, 2.)){
                mg[j][k] = 2;
                pg[j][k] = 0;
            } else {
                    mg[j][k] = 0;
                    pg[j][k] = 0;
                }
            if (j == 0 || j == H+2) {
                mg[j][k] = 2;
                pg[j][k] = 2;
            }
            if (j == 1 || j == H+1) {
                mg[j][k] = 1;
                pg[j][k] = 1;
            }  
        }
    };
    
    
    for (int j = int(1.0/2.*(H+2.) - r)-1; j < int(1.0/2.*(H+2.) + r)+2; j++){
        for (int k = int(1.0/4.*L - r)-1; k < int(1.0/4.*L + r)+2; k++){
            if (mg[j][k] == 0) {
                if (mg[j-1][k] == 2 || mg[j+1][k] == 2 || mg[j][k-1] == 2 || mg[j][k+1] == 2 || mg[j+1][k+1] == 2 || mg[j+1][k-1] == 2 ||
                        mg[j-1][k+1] == 2 || mg[j-1][k-1] == 2) {
                    mg[j][k] = 1;
                }
            }        
        }
    };
    
    /*Visualize the MeshGrid generated if correct*/
    /*
    for (int j = 0; j < H+3; j++) {
        for (int k = 0; k < L+1; k++){
            cout << mg[j][k] << ' ';
        }
        cout << endl;
    }
    for (int j = 0; j < H+3; j++) {
        for (int k = 0; k < L+1; k++){
            cout << pg[j][k] << ' ';
        }
        cout << endl;
    }
    */
    
};



/*Define Lattice Boltzmann related variables/functions:*/
double e[9][2] = {
    {0., 0.},
    {1., 0.},
    {0., 1.},
    {-1., 0.},
    {0., -1.},
    {1., 1.},
    {-1., 1.},
    {-1., -1.},
    {1., -1.}
};


/*to store:
 * dens: density information
 * vel: velocity information
 * f_eq: equilibrium probability information for 9 directions
 * f0 ... f8: probability information for 9 directions; 
 *            - [][][0]: pre-stream
 *            - [][][1]: post-stream
 *            - [][][2]: post-collision equilibrium for this timestep
 * 
 *   */
double dens[n][m];
double vel[n][m][2];
double f_eq[n][m][9];
double f0[n][m][3];
double f1[n][m][3];
double f2[n][m][3];
double f3[n][m][3];
double f4[n][m][3];
double f5[n][m][3];
double f6[n][m][3];
double f7[n][m][3];
double f8[n][m][3];

/*Store velocity in x direction and density from last timestep*/
/*Useful for steady state Poisseuille flow criteria, when calculating relative velocity change*/
double oldVelX[n][m];

/*Store steady state Poisseuille flow velocity in x direction as Inlet fixed velocity for flow past Cylinder*/
double iniVelC[n];

/*Weights for calculating equilibrium probability*/
double w[9] = {4./9, 1./9, 1./9, 1./9, 1./9, 1./36, 1./36, 1./36, 1./36};

/*Calculate macroscopic density from mesoscopic probability f0...f8*/
void density(int s, string str){
    
    if (str == "Poiseuille") {
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for
            for (int j = 1; j < n-1; j++) {
                for (int k = 0; k < m; k++){
                    dens[j][k] = f0[j][k][s] + f1[j][k][s] + f2[j][k][s] + f3[j][k][s] + f4[j][k][s] + 
                            f5[j][k][s] + f6[j][k][s] + f7[j][k][s] + f8[j][k][s];
                }
            }
        }
    }
    
    if (str == "Cylinder") {
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for
            for (int j = 1; j < n-1; j++) {
                for (int k = 0; k < m; k++){
                    if (mg[j][k] == 2) {
                        dens[j][k] = 0;
                    } else {
                        dens[j][k] = f0[j][k][s] + f1[j][k][s] + f2[j][k][s] + f3[j][k][s] + f4[j][k][s] + 
                            f5[j][k][s] + f6[j][k][s] + f7[j][k][s] + f8[j][k][s];
                    }

                }
            }
        }
    }
    
    
};


/*Calculate macroscopic velocity from mesoscopic probability f0...f8 and macroscopic density*/
void velocity(int s, string str){
    if (str == "Poiseuille") {
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for
            for (int j = 1; j < n-1; j++) {
                for (int k = 0; k < m; k++){
                    for (int i = 0; i < 2; i++){
                        vel[j][k][i] = 1./dens[j][k] * c * (f0[j][k][s] * e[0][i] + f1[j][k][s] * e[1][i]
                                + f2[j][k][s] * e[2][i] + f3[j][k][s] * e[3][i] + f4[j][k][s] * e[4][i]
                                + f5[j][k][s] * e[5][i] + f6[j][k][s] * e[6][i] + f7[j][k][s] * e[7][i] 
                                + f8[j][k][s] * e[8][i]);
                    }
                }
            }  
        }    
    }
    
    if (str == "Cylinder") {
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for
            for (int j = 1; j < n-1; j++) {
                for (int k = 0; k < m; k++){
                    for (int i = 0; i < 2; i++){
                        
                        if (mg[j][k] == 2) {
                            vel[j][k][i] = 0;
                        } else {
                            vel[j][k][i] = 1./dens[j][k] * c * (f0[j][k][s] * e[0][i] + f1[j][k][s] * e[1][i]
                                + f2[j][k][s] * e[2][i] + f3[j][k][s] * e[3][i] + f4[j][k][s] * e[4][i]
                                + f5[j][k][s] * e[5][i] + f6[j][k][s] * e[6][i] + f7[j][k][s] * e[7][i] 
                                + f8[j][k][s] * e[8][i]);    
                        }
                    }
                }
            }  
        }       
    }
    
    
    
};

/*Calculate equilibrium probability of 9 directions, based on velocity and density information*/
void feq(string str){
    if (str == "Poiseuille") {
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for
            for (int j = 1; j < n-1; j++) {
                for (int k = 0; k < m; k++){
                    for (int i = 0; i < 9; i++){ 

                        double s = w[i] * (3. * (e[i][0] * vel[j][k][0] + e[i][1] * vel[j][k][1])/c + 
                            9/2.0 * pow((e[i][0] * vel[j][k][0] + e[i][1] * vel[j][k][1]),2.)/pow(c, 2.) - 
                            3/2.0 * (vel[j][k][0] * vel[j][k][0] + vel[j][k][1] * vel[j][k][1])/pow(c, 2.));

                        f_eq[j][k][i] = w[i] * dens[j][k] + dens[j][k] * s;

                    }
                }
            }       
        }
    } 
    
    
    if (str == "Cylinder") {
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for
            for (int j = 1; j < n-1; j++) {
                for (int k = 0; k < m; k++){
                    for (int i = 0; i < 9; i++){ 
                        
                        if (mg[j][k] == 2) {
                            f_eq[j][k][i] = 0;
                        } else {
                            double s = w[i] * (3. * (e[i][0] * vel[j][k][0] + e[i][1] * vel[j][k][1])/c + 
                                9/2.0 * pow((e[i][0] * vel[j][k][0] + e[i][1] * vel[j][k][1]),2)/pow(c, 2.) - 
                                3/2.0 * (vel[j][k][0] * vel[j][k][0] + vel[j][k][1] * vel[j][k][1])/pow(c, 2.));

                            f_eq[j][k][i] = w[i] * dens[j][k] + dens[j][k] * s;
    
                        }
                    }
                }
            }       
        }    
    }   
};

/*Initial condition setup for Poiseuille Flow*/
void initial(){
    #pragma omp parallel num_threads(nt)
    {
        #pragma omp for
        for (int j = 1; j < n-1; j++) {
            for (int k = 0; k < m; k++){

                if (k == 0){
                    dens[j][k] = p0;
                } else if (k == m-1){
                    dens[j][k] = p1;
                } else {
                    dens[j][k] = (p0+p1)/2;
                }

                vel[j][k][0] = 0.;
                vel[j][k][1] = 0.;
            }
        }
    }

    feq("Poiseuille");
    #pragma omp parallel num_threads(nt)
    {
        #pragma omp for
        for (int j = 1; j < n-1; j++) {
            for (int k = 0; k < m; k++){

                f0[j][k][0] = f_eq[j][k][0];
                f1[j][k][0] = f_eq[j][k][1];
                f2[j][k][0] = f_eq[j][k][2];
                f3[j][k][0] = f_eq[j][k][3];
                f4[j][k][0] = f_eq[j][k][4];
                f5[j][k][0] = f_eq[j][k][5];
                f6[j][k][0] = f_eq[j][k][6];
                f7[j][k][0] = f_eq[j][k][7];
                f8[j][k][0] = f_eq[j][k][8];

            }
        }        
    }

};

/*Boundary Conditions*/
/*BC: On-grid Bounce Back for upper and lower walls, and the cylinder if appears*/
/*On-grid Bounce Back or Update normally: This handles all grids (j, k) except for the 
 *Inlet, Outlet and 4 corners. 
 *(j, k) should also not be the dummy walls padded with 2 at mg/pg: [0][:] and [n-1][:]*/
void onGrid(string s, int j, int k){
    if (s == "Poiseuille"){
        if (pg[j][k] == 1){
            f0[j][k][1] = f0[j][k][0];
            if (pg[j-1][k] == 2){
                f4[j][k][1] = f2[j][k][0];
            } else {
                f4[j][k][1] = f4[j-1][k][0];
            }
            
            if (pg[j][k-1] == 2){
                f1[j][k][1] = f3[j][k][0];
            } else {
                f1[j][k][1] = f1[j][k-1][0];
            }
            
            if (pg[j+1][k] == 2){
                f2[j][k][1] = f4[j][k][0];
            } else {
                f2[j][k][1] = f2[j+1][k][0];
            }
            
            
            if (pg[j][k+1] == 2){
                f3[j][k][1] = f1[j][k][0];
            } else {
                f3[j][k][1] = f3[j][k+1][0];
            }
            
            if (pg[j-1][k+1] == 2){
                f7[j][k][1] = f5[j][k][0];
            } else {
                f7[j][k][1] = f7[j-1][k+1][0];
            }
            
            if (pg[j-1][k-1] == 2){
                f8[j][k][1] = f6[j][k][0];
            } else {
                f8[j][k][1] = f8[j-1][k-1][0];
            }
            
            if (pg[j+1][k-1] == 2){
                f5[j][k][1] = f7[j][k][0];
            } else {
                f5[j][k][1] = f5[j+1][k-1][0];
            }
            
            if (pg[j+1][k+1] == 2){
                f6[j][k][1] = f8[j][k][0];
            } else {
                f6[j][k][1] = f6[j+1][k+1][0];
            }
            
        }
        
        if (pg[j][k] == 0){
            f0[j][k][1] = f0[j][k][0];
            f1[j][k][1] = f1[j][k-1][0];
            f2[j][k][1] = f2[j+1][k][0];
            f3[j][k][1] = f3[j][k+1][0];
            f4[j][k][1] = f4[j-1][k][0];
            f5[j][k][1] = f5[j+1][k-1][0];
            f6[j][k][1] = f6[j+1][k+1][0];
            f7[j][k][1] = f7[j-1][k+1][0];
            f8[j][k][1] = f8[j-1][k-1][0];
        }
        
        
    }
    
    if (s == "Cylinder"){
        if (mg[j][k] == 1){
            f0[j][k][1] = f0[j][k][0];
            if (mg[j-1][k] == 2){
                f4[j][k][1] = f2[j][k][0];
            } else {
                f4[j][k][1] = f4[j-1][k][0];
            }
            
            if (mg[j][k-1] == 2){
                f1[j][k][1] = f3[j][k][0];
            } else {
                f1[j][k][1] = f1[j][k-1][0];
            }
            
            if (mg[j+1][k] == 2){
                f2[j][k][1] = f4[j][k][0];
            } else {
                f2[j][k][1] = f2[j+1][k][0];
            }
            
            
            if (mg[j][k+1] == 2){
                f3[j][k][1] = f1[j][k][0];
            } else {
                f3[j][k][1] = f3[j][k+1][0];
            }
            
            if (mg[j-1][k+1] == 2){
                f7[j][k][1] = f5[j][k][0];
            } else {
                f7[j][k][1] = f7[j-1][k+1][0];
            }
            
            if (mg[j-1][k-1] == 2){
                f8[j][k][1] = f6[j][k][0];
            } else {
                f8[j][k][1] = f8[j-1][k-1][0];
            }
            
            if (mg[j+1][k-1] == 2){
                f5[j][k][1] = f7[j][k][0];
            } else {
                f5[j][k][1] = f5[j+1][k-1][0];
            }
            
            if (mg[j+1][k+1] == 2){
                f6[j][k][1] = f8[j][k][0];
            } else {
                f6[j][k][1] = f6[j+1][k+1][0];
            }
            
        }
        if (mg[j][k] == 0){
            f0[j][k][1] = f0[j][k][0];
            f1[j][k][1] = f1[j][k-1][0];
            f2[j][k][1] = f2[j+1][k][0];
            f3[j][k][1] = f3[j][k+1][0];
            f4[j][k][1] = f4[j-1][k][0];
            f5[j][k][1] = f5[j+1][k-1][0];
            f6[j][k][1] = f6[j+1][k+1][0];
            f7[j][k][1] = f7[j-1][k+1][0];
            f8[j][k][1] = f8[j-1][k-1][0];
        }

    }

};

/*Zou-He BCs for Inlet & Outlet*/
void ZouHe(string str1, string str2, int j, int k){
    
    if (str1 == "Poiseuille"){
        /*p0, vy = 0*/
        if (str2 == "inlet"){
            f0[j][k][1] = f0[j][k][0];
            f2[j][k][1] = f2[j+1][k][0];
            f6[j][k][1] = f6[j+1][k+1][0];
            f3[j][k][1] = f3[j][k+1][0];
            f7[j][k][1] = f7[j-1][k+1][0];
            f4[j][k][1] = f4[j-1][k][0];
            
            double vx = c * (1.- 1./p0 * (f0[j][k][1] + f2[j][k][1] + 
                    f4[j][k][1] + 2. * (f3[j][k][1] + f6[j][k][1] + 
                    f7[j][k][1])));
            
            double s1 = 1./9 * (3. * (e[1][0] * vx) /c + 9./2 * pow((e[1][0] * vx), 2.) / pow(c, 2.)
                        - 3./2 * (vx * vx)/pow(c, 2.));
            double s3 = 1./9 * (3. * (e[3][0] * vx) /c + 9./2 * pow((e[3][0] * vx), 2.) / pow(c, 2.)
                        - 3./2 * (vx * vx)/pow(c, 2.));
            double f_eq1 =  1./9 * p0 + p0 * s1;
            double f_eq3 =  1./9 * p0 + p0 * s3;

            f1[j][k][1] = f3[j][k][1] + f_eq1 - f_eq3;
            f5[j][k][1] = 1./2 * (p0 * vx/c - f1[j][k][1] - f2[j][k][1] + 
                           f3[j][k][1] + f4[j][k][1] + 2*f7[j][k][1]);
            f8[j][k][1] = 1./2 * (p0 * vx/c - f1[j][k][1] + f2[j][k][1] + 
                           f3[j][k][1] - f4[j][k][1] + 2*f6[j][k][1]);
        }
        
        /*p1, vy = 0*/
        if (str2 == "outlet"){
            f0[j][k][1] = f0[j][k][0];
            f2[j][k][1] = f2[j+1][k][0];
            f5[j][k][1] = f5[j+1][k-1][0];
            f1[j][k][1] = f1[j][k-1][0];
            f8[j][k][1] = f8[j-1][k-1][0];
            f4[j][k][1] = f4[j-1][k][0];
            
            double vx = c * (-1. +  1./p1 * (f0[j][k][1] + f2[j][k][1] + 
                 f4[j][k][1] + 2. * (f1[j][k][1] + f5[j][k][1] + 
                   f8[j][k][1])));
            double s1 = 1./9 * (3. * (e[1][0] * vx) /c + 9./2 * pow((e[1][0] * vx), 2.) / pow(c, 2.)
                        - 3./2 * (vx * vx)/pow(c, 2.));
            double s3 = 1./9 * (3. * (e[3][0] * vx) /c + 9./2 * pow((e[3][0] * vx), 2.) / pow(c, 2.)
                        - 3./2 * (vx * vx)/pow(c, 2.));
            double f_eq1 =  1./9 * p1 + p1 * s1;
            double f_eq3 =  1./9 * p1 + p1 * s3;

            f3[j][k][1] = f1[j][k][1] + f_eq3 - f_eq1;
            f6[j][k][1] = 1./2 * (-p1*vx/c + f1[j][k][1] - f2[j][k][1] - 
                           f3[j][k][1] + f4[j][k][1] + 2*f8[j][k][1]);
            f7[j][k][1] = 1./2 * (-p1 * vx/c + f1[j][k][1] + f2[j][k][1] - 
                           f3[j][k][1] - f4[j][k][1] + 2*f5[j][k][1]);
   
        }    
    }
    
    if (str1 == "Cylinder"){
        
        /*vx, vy = 0*/
        if (str2 == "inlet"){
            f0[j][k][1] = f0[j][k][0];
            f2[j][k][1] = f2[j+1][k][0];
            f6[j][k][1] = f6[j+1][k+1][0];
            f3[j][k][1] = f3[j][k+1][0];
            f7[j][k][1] = f7[j-1][k+1][0];
            f4[j][k][1] = f4[j-1][k][0];
            
            
            double vx = iniVelC[j];
            double p0C = (2.*(f3[j][k][1] + f6[j][k][1] + f7[j][k][1]) + f0[j][k][1] 
                        + f2[j][k][1] + f4[j][k][1])/(1.-vx/c);
            double s1 = 1./9 * (3. * (e[1][0] * vx) /c + 9./2 * pow((e[1][0] * vx), 2.) / pow(c, 2.)
                        - 3./2 * (vx * vx)/pow(c, 2.));
            double s3 = 1./9 * (3. * (e[3][0] * vx) /c + 9./2 * pow((e[3][0] * vx), 2.) / pow(c, 2.)
                        - 3./2 * (vx * vx)/pow(c, 2.));
            double f_eq1 =  1./9 * p0C + p0C * s1;
            double f_eq3 =  1./9 * p0C + p0C * s3;

            f1[j][k][1] = f3[j][k][1] + f_eq1 - f_eq3;
            f5[j][k][1] = 1./2 * (p0C * vx/c - f1[j][k][1] - f2[j][k][1] + 
                           f3[j][k][1] + f4[j][k][1] + 2*f7[j][k][1]);
            f8[j][k][1] = 1./2 * (p0C * vx/c - f1[j][k][1] + f2[j][k][1] + 
                           f3[j][k][1] - f4[j][k][1] + 2*f6[j][k][1]);
        }
        
        /*p1, vy = 0*/
        if (str2 == "outlet"){
            f0[j][k][1] = f0[j][k][0];
            f2[j][k][1] = f2[j+1][k][0];
            f5[j][k][1] = f5[j+1][k-1][0];
            f1[j][k][1] = f1[j][k-1][0];
            f8[j][k][1] = f8[j-1][k-1][0];
            f4[j][k][1] = f4[j-1][k][0];
            
            double vx = c * (-1. +  1./p1 * (f0[j][k][1] + f2[j][k][1] + 
                 f4[j][k][1] + 2. * (f1[j][k][1] + f5[j][k][1] + 
                   f8[j][k][1])));
            double s1 = 1./9 * (3. * (e[1][0] * vx) /c + 9./2 * pow((e[1][0] * vx), 2.) / pow(c, 2.)
                        - 3./2 * (vx * vx)/pow(c, 2.));
            double s3 = 1./9 * (3. * (e[3][0] * vx) /c + 9./2 * pow((e[3][0] * vx), 2.) / pow(c, 2.)
                        - 3./2 * (vx * vx)/pow(c, 2.));
            double f_eq1 =  1./9 * p1 + p1 * s1;
            double f_eq3 =  1./9 * p1 + p1 * s3;

            f3[j][k][1] = f1[j][k][1] + f_eq3 - f_eq1;
            f6[j][k][1] = 1./2 * (-p1*vx/c + f1[j][k][1] - f2[j][k][1] - 
                           f3[j][k][1] + f4[j][k][1] + 2*f8[j][k][1]);
            f7[j][k][1] = 1./2 * (-p1 * vx/c + f1[j][k][1] + f2[j][k][1] - 
                           f3[j][k][1] - f4[j][k][1] + 2*f5[j][k][1]);
        }  
    }      
};




/*BCs for Corners*/
void corner(string s, int j, int k) {
    
    /*using Outlet fix pressure p1 information*/
    if (j == 1 && k == m-1) {
        f0[j][k][1] = f0[j][k][0];
        f2[j][k][1] = f2[j+1][k][0];
        f5[j][k][1] = f5[j+1][k-1][0];
        f1[j][k][1] = f1[j][k-1][0];

        f3[j][k][1] = f1[j][k][1];
        f7[j][k][1] = f5[j][k][1];
        f4[j][k][1] = f2[j][k][1];

        f6[j][k][1] = 1./2 * (p1 - f0[j][k][1] - f1[j][k][1] - 
                      f2[j][k][1] - f3[j][k][1] - f4[j][k][1] - 
                      f5[j][k][1] - f7[j][k][1]);
        f8[j][k][1] = f6[j][k][1];
        
    }
    
    if (j == n-2 && k == m-1) {
        f0[j][k][1] = f0[j][k][0];
        f1[j][k][1] = f1[j][k-1][0];
        f8[j][k][1] = f8[j-1][k-1][0];
        f4[j][k][1] = f4[j-1][k][0];

        f3[j][k][1] = f1[j][k][1];
        f6[j][k][1] = f8[j][k][1];
        f2[j][k][1] = f4[j][k][1];

        f5[j][k][1] = 1./2 * (p1 - f0[j][k][1] - f1[j][k][1] - 
                      f2[j][k][1] - f3[j][k][1] - f4[j][k][1] - 
                      f6[j][k][1] - f8[j][k][1]);
        f7[j][k][1] = f5[j][k][1];
    }
    
    if (s == "Poiseuille") {
        
        /*using Inlet fixed pressure p0 information*/
        if (j == 1 && k == 0) {
            f0[j][k][1] = f0[j][k][0];
            f2[j][k][1] = f2[j+1][k][0];
            f6[j][k][1] = f6[j+1][k+1][0];
            f3[j][k][1] = f3[j][k+1][0];

            f1[j][k][1] = f3[j][k][1];
            f8[j][k][1] = f6[j][k][1];
            f4[j][k][1] = f2[j][k][1];

            f5[j][k][1] = 1./2 * (p0 - f0[j][k][1] - f1[j][k][1] - 
                          f2[j][k][1] - f3[j][k][1] - f4[j][k][1] - 
                          f6[j][k][1] - f8[j][k][1]);
            f7[j][k][1] = f5[j][k][1];
        }

        if (j == n-2 && k == 0) {
            f0[j][k][1] = f0[j][k][0];
            f3[j][k][1] = f3[j][k+1][0];
            f7[j][k][1] = f7[j-1][k+1][0];
            f4[j][k][1] = f4[j-1][k][0];

            f1[j][k][1] = f3[j][k][1];
            f5[j][k][1] = f7[j][k][1];
            f2[j][k][1] = f4[j][k][1];

            f6[j][k][1] = 1./2 * (p0 - f0[j][k][1] - f1[j][k][1] - 
                          f2[j][k][1] - f3[j][k][1] - f4[j][k][1] - 
                          f5[j][k][1] - f7[j][k][1]);
            f8[j][k][1] = f6[j][k][1]; 
        }
        
    }
    
    if (s == "Cylinder") {
        
        /*Inlet fixed velocity, but pressure unknown*/
        if (j == 1 && k == 0) {
            f0[j][k][1] = f0[j][k][0];
            f2[j][k][1] = f2[j+1][k][0];
            f6[j][k][1] = f6[j+1][k+1][0];
            f3[j][k][1] = f3[j][k+1][0];

            f1[j][k][1] = f3[j][k][1];
            f8[j][k][1] = f6[j][k][1];
            f4[j][k][1] = f2[j][k][1];
            
            /*Extrapolate pressure information from nearest point at boundary*/
            double pjk = f0[j+1][k][1] + f1[j+1][k][1] + f2[j+1][k][1] + f3[j+1][k][1] + f4[j+1][k][1] + f5[j+1][k][1] + f6[j+1][k][1] + f7[j+1][k][1] + f8[j+1][k][1];
            f5[j][k][1] = 1./2 * (pjk - f0[j][k][1] - f1[j][k][1] - 
                          f2[j][k][1] - f3[j][k][1] - f4[j][k][1] - 
                          f6[j][k][1] - f8[j][k][1]);
            f7[j][k][1] = f5[j][k][1];
        }

        if (j == n-2 && k == 0) {
            f0[j][k][1] = f0[j][k][0];
            f3[j][k][1] = f3[j][k+1][0];
            f7[j][k][1] = f7[j-1][k+1][0];
            f4[j][k][1] = f4[j-1][k][0];

            f1[j][k][1] = f3[j][k][1];
            f5[j][k][1] = f7[j][k][1];
            f2[j][k][1] = f4[j][k][1];
            
            /*Extrapolate pressure information from nearest point at boundary*/
            double pjk = f0[j-1][k][1] + f1[j-1][k][1] + f2[j-1][k][1] + f3[j-1][k][1] + f4[j-1][k][1] + f5[j-1][k][1] + f6[j-1][k][1] + f7[j-1][k][1] + f8[j-1][k][1];
            f6[j][k][1] = 1./2 * (pjk - f0[j][k][1] - f1[j][k][1] - 
                          f2[j][k][1] - f3[j][k][1] - f4[j][k][1] - 
                          f5[j][k][1] - f7[j][k][1]);
            f8[j][k][1] = f6[j][k][1]; 
        }

    }
    
}




















int main ()
{   
    /*Create pg for Poiseuille Flow and mg for cylinder immersed flow*/
    createGrid();
    
/*=========================================================================================================*/
    /*Poiseuille Flow Simulation*/
    /*First, simulate Poiseuille Flow till steady state*/
    initial();  
    
    int ct = 1;
    double relVel = 1.;
    
    /*Steady State Criteria; number of timesteps*/
    while (relVel > 5.0 * pow(10, -9.)){   

        /*Streaming*/
        /*Exclude the 2's over upper & lower walls*/
        /*Exclude the four corners*/
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for
            for (int j = 1; j < n-1; j++) {
                for (int k = 0; k < m; k++){

                    if (k != 0 && k != (m-1)){
                        /*Update all points expect inlet, outlet and corners*/
                        onGrid("Poiseuille", j, k);

                    } 

                    if (k == 0 && j != 1 && j != n-2){
                        /*inlet BC*/
                        ZouHe("Poiseuille", "inlet", j, k);

                    } 

                    if (k == (m-1) && j != 1 && j != n-2) {
                        /*outlet BC*/
                        ZouHe("Poiseuille", "outlet", j, k);

                    } 
                }
            }
        }
        
        
        /*four corners BC streaming*/
        corner("Poiseuille", 1, m-1);
        corner("Poiseuille", n-2, m-1);
        corner("Poiseuille", 1, 0);
        corner("Poiseuille", n-2, 0);
        
        
        /*Store x direction velocity information before changing it*/
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for
            for (int j = 1; j < n-1; j++) {
                for (int k = 0; k < m; k++){
                    oldVelX[j][k] = vel[j][k][0];
                }
            }
        }

        /*update post streaming density and velocity and f_eq*/
        density(1, "Poiseuille");
        velocity(1, "Poiseuille");
        feq("Poiseuille");

        /*Collision*/
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for
            for (int j = 1; j < n-1; j++) {
                for (int k = 0; k < m; k++){
                    
                    /*Update post-collision probability as the pre-streaming probability for next timestep*/
                    f0[j][k][0] = f0[j][k][1] - 1./tau * (f0[j][k][1] - f_eq[j][k][0]);
                    f1[j][k][0] = f1[j][k][1] - 1./tau * (f1[j][k][1] - f_eq[j][k][1]);
                    f2[j][k][0] = f2[j][k][1] - 1./tau * (f2[j][k][1] - f_eq[j][k][2]);
                    f3[j][k][0] = f3[j][k][1] - 1./tau * (f3[j][k][1] - f_eq[j][k][3]);
                    f4[j][k][0] = f4[j][k][1] - 1./tau * (f4[j][k][1] - f_eq[j][k][4]);
                    f5[j][k][0] = f5[j][k][1] - 1./tau * (f5[j][k][1] - f_eq[j][k][5]);
                    f6[j][k][0] = f6[j][k][1] - 1./tau * (f6[j][k][1] - f_eq[j][k][6]);
                    f7[j][k][0] = f7[j][k][1] - 1./tau * (f7[j][k][1] - f_eq[j][k][7]);
                    f8[j][k][0] = f8[j][k][1] - 1./tau * (f8[j][k][1] - f_eq[j][k][8]);
                    
                    /*Storing post-collision equilibrium for this timestep in fi[][][2]*/
                    f0[j][k][2] = f_eq[j][k][0];
                    f1[j][k][2] = f_eq[j][k][1];
                    f2[j][k][2] = f_eq[j][k][2];
                    f3[j][k][2] = f_eq[j][k][3];
                    f4[j][k][2] = f_eq[j][k][4];
                    f5[j][k][2] = f_eq[j][k][5];
                    f6[j][k][2] = f_eq[j][k][6];
                    f7[j][k][2] = f_eq[j][k][7];
                    f8[j][k][2] = f_eq[j][k][8];
   
                }
            }
        }
        
        density(2, "Poiseuille");
        velocity(2, "Poiseuille");
        
        
        /*Calculate relative velocity change*/
        /*Save density and velocity information at this timestep*/ 
        double relVelN = 0.0;
        double relVelD = 0.0; 
        ofstream out_dens ("/home/jiayinlu/Desktop/Kay/LB/Poisseuille/Density/textfile/density"+ to_string(ct) +".txt");
        ofstream out_vel ("/home/jiayinlu/Desktop/Kay/LB/Poisseuille/Velocity/textfile/velocity"+ to_string(ct) +".txt");
        
        for (int j = 1; j < n-1; j++) {
            for (int k = 0; k < m; k++){
                
                relVelN = relVelN + abs(vel[j][k][0] - oldVelX[j][k]);
                relVelD = relVelD + abs(vel[j][k][0]);
                
                out_dens << dens[j][k] << " " ;
                
                for (int i = 0; i < 2; i++){
                    
                    out_vel << vel[j][k][i] << " " ;
                    
                }
                
            }
            out_dens << endl;
            out_vel << endl;   
        }
        
        relVel = relVelN/relVelD;
        ct ++;
        
    };
    
    
    /*Store steady Poisseuille Flow x velocity information for Fixed Velocity Inlet in Cylinder case*/
    #pragma omp parallel num_threads(nt)
    {
        #pragma omp for
        for (int j = 1; j < n-1; j++) {
            iniVelC[j] = vel[j][m-1][0];
        }
    };
 /*===============================================================================================================*/
    
    /*Flow past cylinder*/
    /*Set up initial conditions for Flow Past Cylinder; Use the above steady state Poisseulle Flow profile*/
    #pragma omp parallel num_threads(nt)
    {
        #pragma omp for
        for (int j = 1; j < n-1; j++) {
            for (int k = 0; k < m; k++){

                f0[j][k][0] = f_eq[j][k][0];
                f1[j][k][0] = f_eq[j][k][1];
                f2[j][k][0] = f_eq[j][k][2];
                f3[j][k][0] = f_eq[j][k][3];
                f4[j][k][0] = f_eq[j][k][4];
                f5[j][k][0] = f_eq[j][k][5];
                f6[j][k][0] = f_eq[j][k][6];
                f7[j][k][0] = f_eq[j][k][7];
                f8[j][k][0] = f_eq[j][k][8];

            }
        }        
    };
    
    int ct2 = 1;
    
    /*Determine number of timesteps*/
    while (ct2 < 100000) {
        
        /*Streaming*/
        /*Exclude the 2's over upper & lower walls*/
        /*Exclude the four corners*/
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for
            for (int j = 1; j < n-1; j++) {
                for (int k = 0; k < m; k++){

                    if (k != 0 && k != (m-1)){
                        /*Update all points expect inlet, outlet and corners*/
                        onGrid("Cylinder", j, k);

                    } 

                    if (k == 0 && j != 1 && j != n-2){
                        /*inlet BC*/
                        ZouHe("Cylinder", "inlet", j, k);

                    } 

                    if (k == (m-1) && j != 1 && j != n-2) {
                        /*outlet BC*/
                        ZouHe("Cylinder", "outlet", j, k);

                    } 
                }
            }
        }
        
        
        /*four corners BC streaming*/
        corner("Cylinder", 1, m-1);
        corner("Cylinder", n-2, m-1);
        corner("Cylinder", 1, 0);
        corner("Cylinder", n-2, 0);
        
        
        /*update post streaming density and velocity and f_eq*/
        density(1, "Cylinder");
        velocity(1, "Cylinder");
        feq("Cylinder");

        /*Collision*/
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for
            for (int j = 1; j < n-1; j++) {
                for (int k = 0; k < m; k++){
                    
                    /*Update post-collision probability as the pre-streaming probability for next timestep*/
                    f0[j][k][0] = f0[j][k][1] - 1./tau * (f0[j][k][1] - f_eq[j][k][0]);
                    f1[j][k][0] = f1[j][k][1] - 1./tau * (f1[j][k][1] - f_eq[j][k][1]);
                    f2[j][k][0] = f2[j][k][1] - 1./tau * (f2[j][k][1] - f_eq[j][k][2]);
                    f3[j][k][0] = f3[j][k][1] - 1./tau * (f3[j][k][1] - f_eq[j][k][3]);
                    f4[j][k][0] = f4[j][k][1] - 1./tau * (f4[j][k][1] - f_eq[j][k][4]);
                    f5[j][k][0] = f5[j][k][1] - 1./tau * (f5[j][k][1] - f_eq[j][k][5]);
                    f6[j][k][0] = f6[j][k][1] - 1./tau * (f6[j][k][1] - f_eq[j][k][6]);
                    f7[j][k][0] = f7[j][k][1] - 1./tau * (f7[j][k][1] - f_eq[j][k][7]);
                    f8[j][k][0] = f8[j][k][1] - 1./tau * (f8[j][k][1] - f_eq[j][k][8]);
                    
                    /*Storing post-collision equilibrium for this timestep in fi[][][2]*/
                    f0[j][k][2] = f_eq[j][k][0];
                    f1[j][k][2] = f_eq[j][k][1];
                    f2[j][k][2] = f_eq[j][k][2];
                    f3[j][k][2] = f_eq[j][k][3];
                    f4[j][k][2] = f_eq[j][k][4];
                    f5[j][k][2] = f_eq[j][k][5];
                    f6[j][k][2] = f_eq[j][k][6];
                    f7[j][k][2] = f_eq[j][k][7];
                    f8[j][k][2] = f_eq[j][k][8];
   
                }
            }
        }
        
        density(2, "Cylinder");
        velocity(2, "Cylinder");
        
        
        /*Save density and velocity information at this timestep*/ 
        ofstream out_dens ("/home/jiayinlu/Desktop/Kay/LB/Cylinder1/Density/textfile/density"+ to_string(ct2) +".txt");
        ofstream out_vel ("/home/jiayinlu/Desktop/Kay/LB/Cylinder1/Velocity/textfile/velocity"+ to_string(ct2) +".txt");
        for (int j = 1; j < n-1; j++) {
            for (int k = 0; k < m; k++){
                
                out_dens << dens[j][k] << " " ;
                
                for (int i = 0; i < 2; i++){
                    
                    out_vel << vel[j][k][i] << " " ;
                    
                }
                
            }
            out_dens << endl;
            out_vel << endl;   
        }
        
        ct2 ++;
        

        
       
    };
    
    

    return 0;
}
