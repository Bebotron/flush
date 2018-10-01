#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
//#include <Eigen>
//#include <KroneckerProduct>

#include "omp.h"
#include "basic_funcs_vec.h"
#include "time.h"

using namespace std;
using namespace Eigen;

void printRunTime(time_t t0, time_t t1) {
    int t_run, d, h, m, s; 
    t_run = difftime(t1, t0);
    d = floor(t_run/(24*3600));
    h = floor(t_run/3600);
    m = floor(t_run/60); m = m%60;
    s = t_run%60;
    cout << "TIME TO RUN:\n"
         << d << "d " 
         << h << "h " 
         << m << "m " 
         << s << "s " << endl;
    return;
}

void outputPlotData(string filename, ArrayXXf& FdataList) {
    ofstream fout;
    fout.open(filename.c_str());
    for(int i = 0; i < FdataList.rows(); i++) {
        for(int j = 0; j < FdataList.cols(); j++) {
            fout << FdataList(i, j) << " ";
        }
        fout << endl;
    }
    fout.close();
    return;
}

void outputPulseData(string filename, ArrayXf& cx, ArrayXf& cy) {
    ofstream fout;
    fout.open(filename.c_str());
    for(int i = 0; i < cx.size(); i++) fout << cx(i) << " ";
    fout << endl;
    for(int i = 0; i < cy.size(); i++) fout << cy(i) << " ";
    fout.close();
    return;
}
/*
void getMatrix(basic_funcs& bf, int Ncycles, int mult, int l, float k, ArrayXf cx, ArrayXf cy, MatrixXcd *rhoList, ArrayXXf FdataList, MatrixXf& matrixM) {
    float collapseOn, collapseOff;
    collapseOn = 1e-3/(k*10); collapseOff = 0.03;

    // basic_funcs bf(collapseOn, collapseOff);
    
    int tp, tf, listLength;
    float dt = 0.1;
    tp = 20; tf = mult*l;
    matrixM = MatrixXf::Zero(6, 6);

    if(Ncycles%2 == 0) listLength = Ncycles*(tp + tf)/(2*dt) + 1;
    else listLength = (tp*ceil(Ncycles/2.0) + tf*floor(Ncycles/2.0))/dt + 1;
    MatrixXcd finalState[6];
    FdataList = ArrayXXf::Zero(2, listLength);
    int t_cyc[] = {tp, tf};
    float dummyF = 0;

    for(int j = 0; j <= 5; j++) {
        for(int i = 0; i <= 5; i++) bf.evolveState(dt, Ncycles, rhoList[i], rhoList[j], t_cyc, cx, cy, 0, 1, FdataList, dummyF, finalState[i]);
        for(int i = 0; i <= 5; i++) matrixM(j, i) = finalState[i](j, j).real();
    }
    return;
}

void optimizeFlushCycle(int mult, int k, int mintf, int maxtf, MatrixXcd *rhoList, ArrayXf cx, ArrayXf cy, ArrayXXf& prob00to10) {
    int size = maxtf - mintf;
    ArrayXXf dummyFdata;
    MatrixXf matrixList[size];
    EigenSolver<MatrixXf> esys[size];
    prob00to10.setZero(2, size);    
    ArrayXf evals;
	VectorXf evecs;
    int evalIndex;
    #pragma omp parallel for
    for(int l = 0; l < size; l++) {
        prob00to10(0, l) = mult*(l + mintf);
        getMatrix(2, mult, l + mintf, k, cx, cy, rhoList, dummyFdata, matrixList[l]);
        esys[l].compute(matrixList[l]);
        evals = esys[l].eigenvalues().real();
        evalIndex = 0;
        for(int i = 0; i < evals.size(); i++) if(evals[i] == evals.maxCoeff()) evalIndex = i;
        evecs = esys[l].eigenvectors().col(evalIndex).real();
        prob00to10(1, l) = ((evecs/evecs.sum())(2));
    }
    return;
}
*/
int main() {
    string evolve_file, pulse_file;
    time_t t0, t1;
    int tp, tf, num_states, numFidelities, Ncycles, Ohm, listLength, maxIt;
    float dt, dc, acc, collapseOn, collapseOff, J, F;
    bool flush, checking_min;
    Vector2cd ks0, ks1;
    Matrix2cd rs0, rs1;
    ks0 << 1, 0; ks1 << 0, 1;
    rs0 << 1, 0, 0, 0; rs1 << 0, 0, 0, 1;
    VectorXcd ket000, ket111, ket100, ket010, ket001, ket110, ket101, ket011, k000100;
    MatrixXcd rho000, rho111, rho100, rho010, rho001, rho110, rho101, rho011, finalState;
    VectorXcd kls000[] = {ks0,ks0,ks0,ks0,ks0,ks0};
    VectorXcd kls111[] = {ks1,ks1,ks1,ks0,ks0,ks0};
    VectorXcd kls100[] = {ks1,ks0,ks0,ks0,ks0,ks0};
    VectorXcd kls010[] = {ks0,ks1,ks0,ks0,ks0,ks0};
    VectorXcd kls001[] = {ks0,ks0,ks1,ks0,ks0,ks0};
    VectorXcd kls110[] = {ks1,ks1,ks0,ks0,ks0,ks0};
    VectorXcd kls101[] = {ks1,ks0,ks1,ks0,ks0,ks0};
    VectorXcd kls011[] = {ks0,ks1,ks1,ks0,ks0,ks0};
    VectorXcd kls000100[] = {ks0,ks0,ks0,ks1,ks0,ks0};
    MatrixXcd rls000[] = {rs0,rs0,rs0,rs0,rs0,rs0};
    MatrixXcd rls111[] = {rs1,rs1,rs1,rs0,rs0,rs0};
    MatrixXcd rls100[] = {rs1,rs0,rs0,rs0,rs0,rs0};
    MatrixXcd rls010[] = {rs0,rs1,rs0,rs0,rs0,rs0};
    MatrixXcd rls001[] = {rs0,rs0,rs1,rs0,rs0,rs0};
    MatrixXcd rls110[] = {rs1,rs1,rs0,rs0,rs0,rs0};
    MatrixXcd rls101[] = {rs1,rs0,rs1,rs0,rs0,rs0};
    MatrixXcd rls011[] = {rs0,rs1,rs1,rs0,rs0,rs0};
    num_states = 6;

    ArrayXf cx(20), cy(20);
    cx.setZero(); cy.setZero();
    tp = 40; tf = 100; Ncycles = 20; numFidelities = 3;
    dt = 0.01; dc = 1e-6; acc = 1e-5;
    collapseOn = 1e-3/(2*10); collapseOff = 0.03; J = 0.02;
    flush = 0; checking_min = 1;

    ArrayXf fidelities(numFidelities + 1);
    fidelities.setZero();

    basic_funcs_vec bf(collapseOn, collapseOff, J);

    ket000 = bf.tensor_vec(kls000, num_states);
    ket111 = bf.tensor_vec(kls111, num_states);
    ket100 = bf.tensor_vec(kls100, num_states);
    ket010 = bf.tensor_vec(kls010, num_states);
    ket001 = bf.tensor_vec(kls001, num_states);
    ket110 = bf.tensor_vec(kls110, num_states);
    ket101 = bf.tensor_vec(kls101, num_states);
    ket011 = bf.tensor_vec(kls011, num_states);
    k000100 = bf.tensor_vec(kls000100, num_states);
    rho000 = bf.tensor(rls000, num_states);
    rho111 = bf.tensor(rls111, num_states);
    rho100 = bf.tensor(rls100, num_states);
    rho010 = bf.tensor(rls010, num_states);
    rho001 = bf.tensor(rls001, num_states);
    rho110 = bf.tensor(rls110, num_states);
    rho101 = bf.tensor(rls101, num_states);
    rho011 = bf.tensor(rls011, num_states);

    // MatrixXcd rhoList[8] = {rho000, rho100, rho010, rho001, rho110, rho101, rho011, rho111};

    maxIt = 40000;
    if(checking_min) {
        evolve_file = "./tp=" + to_string(tp) + "_tf=" + to_string(tf) + "_numF=" + to_string(numFidelities) + "_min.dat";
        pulse_file = "./tp=" + to_string(tp) + "_numF=" + to_string(numFidelities) + "_min.pls";
    }
    else {
        evolve_file = "./tp=" + to_string(tp) + "_tf=" + to_string(tf) + "_numF=" + to_string(numFidelities) + ".dat";
        pulse_file = "./tp=" + to_string(tp) + "_numF=" + to_string(numFidelities) + ".pls";
    }
    ArrayXXf dataList;
    // if(flush) evolve_file = "./outFiles/yes_coupling.dat";
    // else evolve_file = "./outFiles/no_coupling.dat";

    /*
    evolve_file = "./outFiles/outputF" + to_string(numFidelities) + "_" + to_string(tp);
    if(checking_min) evolve_file += "_min";
    pulse_file = evolve_file + ".pls";
    evolve_file += ".dat";
    */

    cx << 0.00829447, -3.51667e-06, -3.016e-05, -1.31726e-05, -1.77622e-05, -8.76188e-06, -8.64267e-06, -1.18613e-05, -1.23978e-05, -1.80602e-05, -9.94802e-05, -0.0001598, -0.000194311, -0.000140548, -2.31266e-05, -1.57952e-05, -0.00126237, -2.72393e-05, 0.00196034, -2.98023e-07;
    cy << 0.00158238, -8.40425e-06, 1.2517e-05, 1.3113e-06, 1.11461e-05, 4.29153e-06, 9.77516e-06, -3.99351e-06, -5.60284e-06, -1.58548e-05, -9.67979e-05, -0.000161111, -0.000192523, -0.000142634, -1.41859e-05, -2.21729e-05, 0.00124627, 1.43051e-06, -0.00196636, -1.35899e-05;
    // cx[0] = 0.061685; cy[0] = 0.05;
    // cx[0] = 0.01;
    

    time(&t0);

    cout << "**************** ENTERING TIMED SECTION *****************" << endl;
    int t_cyc[] = {tp, tf};
    Ohm = 0;

    // optimize pulse with minimum Fidelity tracking
    // bf.optimizePulse(tp, dt, maxIt, dc, acc, cx, cy, ket000, ket100, ket010, k000100, fidelities, dataList, numFidelities, checking_min);

    // bf.getFidelity(cx, cy, tp, dt, ket000, ket100, ket010, r100100, r100NNN, r010NNN, numFidelities, fidelities, dataList, checking_min);
    

    // cout << "FIDELITY " << (rho111*rho111.adjoint()).cwiseAbs().trace() << endl;
    bf.evolveState(dt, Ncycles, rho111, rho111, tp, tf, cx, cy, Ohm, flush, dataList, F, finalState);
    outputPlotData(evolve_file, dataList);
    // outputPulseData(pulse_file, cx, cy);

    // ArrayXXf prob00to10[5];
    // ArrayXf optVal[5];
    // optVal.setZero();    
    // for(int k = 1; k < 6; k++) {
    //     optimizeFlushCycle(10, k, 1, 13, rhoList, cx, cy, prob00to10[k - 1]);
    //     optVal[k - 1] = prob00to10[k - 1].rowwise().maxCoeff()(1);
    // }
    
    time(&t1);

    // for(int k = 0; k < 5; k++) cout << optVal[k] << ":\n" << prob00to10[k] << endl << endl;
    printRunTime(t0, t1);
    
    return 0;
}
