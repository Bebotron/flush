#include "basic_funcs_vec.h"

basic_funcs_vec::basic_funcs_vec(float cOn, float cOff, float J) {
    int num_ops = 6;
    I = Matrix2cd::Identity();
    MatrixXcd lsI[] = {I,I,I,I,I,I}; Eye = tensor(lsI, num_ops);
    a << 0, 1, 0, 0; X << 0, 1, 1, 0; Z << 1, 0, 0, -1; Y << 0, IM, -IM, 0;
    MatrixXcd X1List[] = {X,I,I,I,I,I}; MatrixXcd X1SList[] = {I,I,I,X,I,I};
    MatrixXcd Y1List[] = {Y,I,I,I,I,I}; MatrixXcd Y1SList[] = {I,I,I,Y,I,I};
    MatrixXcd Z1List[] = {Z,I,I,I,I,I}; MatrixXcd Z1SList[] = {I,I,I,Z,I,I};
    MatrixXcd X2List[] = {I,X,I,I,I,I}; MatrixXcd X2SList[] = {I,I,I,I,X,I};
    MatrixXcd Y2List[] = {I,Y,I,I,I,I}; MatrixXcd Y2SList[] = {I,I,I,I,Y,I};
    MatrixXcd Z2List[] = {I,Z,I,I,I,I}; MatrixXcd Z2SList[] = {I,I,I,I,Z,I};
    MatrixXcd X3List[] = {I,I,X,I,I,I}; MatrixXcd X3SList[] = {I,I,I,I,I,X};
    MatrixXcd Y3List[] = {I,I,Y,I,I,I}; MatrixXcd Y3SList[] = {I,I,I,I,I,Y};
    MatrixXcd Z3List[] = {I,I,Z,I,I,I}; MatrixXcd Z3SList[] = {I,I,I,I,I,Z};
    MatrixXcd a1List[] = {I,I,I,a,I,I};
    MatrixXcd a2List[] = {I,I,I,I,a,I};
    MatrixXcd a3List[] = {I,I,I,I,I,a};

    X1 = tensor(X1List, num_ops); X1S = tensor(X1SList, num_ops);
    Y1 = tensor(Y1List, num_ops); Y1S = tensor(Y1SList, num_ops);
    Z1 = tensor(Z1List, num_ops); Z1S = tensor(Z1SList, num_ops);
    X2 = tensor(X2List, num_ops); X2S = tensor(X2SList, num_ops);
    Y2 = tensor(Y2List, num_ops); Y2S = tensor(Y2SList, num_ops);
    Z2 = tensor(Z2List, num_ops); Z2S = tensor(Z2SList, num_ops);
    X3 = tensor(X3List, num_ops); X3S = tensor(X3SList, num_ops);
    Y3 = tensor(Y3List, num_ops); Y3S = tensor(Y3SList, num_ops);
    Z3 = tensor(Z3List, num_ops); Z3S = tensor(Z3SList, num_ops);
    a1 = tensor(a1List, num_ops); a1d = a1.adjoint();
    a2 = tensor(a2List, num_ops); a2d = a2.adjoint();
    a3 = tensor(a3List, num_ops); a3d = a3.adjoint();

    HP = -J*(Z1*Z2 + Z2*Z3 + Z1*Z3);
    HS = -2*J*(Z1S + Z2S + Z3S);
    HX1 = X1*X1S; HX2 = X2*X2S; HX3 = X3*X3S;
    HY1 = Y1*Y1S; HY2 = Y2*Y2S; HY3 = Y3*Y3S;

    collapseOn = cOn; collapseOff = cOff;
}

basic_funcs_vec::~basic_funcs_vec() {

}

MatrixXcd basic_funcs_vec::tensor(MatrixXcd* matrix_list, int num_matrices) {
    MatrixXcd output = kroneckerProduct(matrix_list[0], matrix_list[1]).eval();
    for(int i = 2; i < num_matrices; i++) output = kroneckerProduct(output, matrix_list[i]).eval();
    return output;
}

VectorXcd basic_funcs_vec::tensor_vec(VectorXcd* vector_list, int num_vectors) {
    VectorXcd output = kroneckerProduct(vector_list[0], vector_list[1]).eval();
    for(int i = 2; i < num_vectors; i++) output = kroneckerProduct(output, vector_list[i]).eval();
    return output;
}

void basic_funcs_vec::getFidelity(ArrayXf cx, ArrayXf cy, int tp, float dt, VectorXcd& ket000, VectorXcd& ket100, VectorXcd& ket010, VectorXcd& k000100, int numFidelities, ArrayXf& fidelities, ArrayXXf& dataList, bool checking_min) {
    VectorXcd currentState1, currentState2, currentState3;
    MatrixXcd H;
    float F, t;
    int Nmax, Findex, listLength;
    Nmax = cx.size();
    currentState1 = ket100; currentState2 = ket000; currentState3 = ket010;
    t = dt;
    listLength = tp/dt;
    dataList.setZero(4, listLength);
    for(int i = 0; i < listLength; i++) {
        H = HP + pulse(t, tp, cx, Nmax)*HX1
               + pulse(t, tp, cy, Nmax)*HY1 + HS;
        timePropagatorRK4(dt, H, currentState1); currentState1.normalize();
        timePropagatorRK4(dt, H, currentState2); currentState2.normalize();
        timePropagatorRK4(dt, H, currentState3); currentState3.normalize();
        dataList(0, i) = t;
        dataList(1, i) = (currentState1.adjoint()*k000100).squaredNorm();
        dataList(2, i) = (currentState2.adjoint()*ket000).squaredNorm();
        dataList(3, i) = (currentState3.adjoint()*ket010).squaredNorm();
        t += dt;
    }
    fidelities(1) = dataList(1, listLength - 1);
    F = fidelities(1);

    if(checking_min) {
        fidelities(2) = dataList(2, listLength - 1);
        fidelities(3) = dataList.rowwise().minCoeff()(2);
    }
    else {
        Findex = floor((listLength - 1)/(numFidelities - 1));
        for(int f = 1; f < numFidelities; f++) {
            fidelities(f + 1) = dataList(2, f*Findex);
            F *= fidelities(f + 1);
        }
    }
    fidelities(0) = F;
    return;
}

// This currently uses a gradient ascent optimization algorithm. Gotta try to use a BFGS algorithm

void basic_funcs_vec::optimizePulse(float tp, float dt, int maxIt, float dc, float acc, ArrayXf& cx, ArrayXf& cy, VectorXcd& ket000, VectorXcd& ket100, VectorXcd& ket010, VectorXcd& k000100, ArrayXf& fidelities, ArrayXXf& dataList, int numFidelities, bool checking_min) {
    float F, eps;
    int Nmax, it; 
    Nmax = cx.size(); it = 0; eps = dc;
    ArrayXf dFx(Nmax), dFy(Nmax);

    getFidelity(cx, cy, tp, dt, ket000, ket100, ket010, k000100, numFidelities, fidelities, dataList, checking_min);
    F = fidelities(0);
    cout << "=== INITIAL FIDELITIES ===\n" << fidelities << endl;
    while((it < maxIt) && ((1 - F) > acc)) {
        if((1 - F)/acc > 100) eps = dc;
        else if((1 - F)/acc > 10) eps = 0.1*dc;
        else eps = 0.01*dc;
        #pragma omp parallel for
        for(int i = 0; i < Nmax; i++) {
            ArrayXf diff_vec(Nmax), tempFidelities(numFidelities + 1);
            ArrayXXf tempDataList;
            diff_vec.setZero();
            for(int j = 0; j < Nmax; j++) if(i == j) diff_vec(j) = 1;
            getFidelity(cx + eps*diff_vec, cy, tp, dt, ket000, ket100, ket010, k000100, numFidelities, tempFidelities, tempDataList, checking_min);
            dFx(i) = tempFidelities(0) - F;
            getFidelity(cx, cy + eps*diff_vec, tp, dt, ket000, ket100, ket010, k000100, numFidelities, tempFidelities, tempDataList, checking_min);
            dFy(i) = tempFidelities(0) - F;
        }
        cx += dFx; cy += dFy;
        getFidelity(cx, cy, tp, dt, ket000, ket100, ket010, k000100, numFidelities, fidelities, dataList, checking_min);
        F = fidelities(0);
        it++;
    }
    cout << "=== FINAL FIDELITIES ===\n" << fidelities << endl;
    cout << "=== NUM OF ITERATIONS ===\n" << it << endl;
    return;
}
/*
void basic_funcs_vec::BFGSoptimize(float tp, float dt, int maxIt, float dc, float acc, ArrayXf& cx, ArrayXf& cy, VectorXcd& ket000, VectorXcd& ket100, VectorXcd& ket010, VectorXcd& k000100, ArrayXf& fidelities, ArrayXXf& dataList, int numFidelities, bool checking_min) {
    float F, eps;
    int Nmax, it; 
    Nmax = cx.size(); it = 0; eps = dc;
    MatrixXf B; B = Eye;
    VectorXf x(2*Nmax), p(2*Nmax), grad(2*Nmax), y(2*Nmax);
    ArrayXf dFx(Nmax), dFy(Nmax);

    x << cx, cy;

    getFidelity(x.head(Nmax), x.tail(Nmax), tp, dt, ket000, ket100, ket010, k000100, numFidelities, fidelities, dataList, checking_min);
    F = fidelities(0);
    cout << "=== INITIAL FIDELITIES ===\n" << fidelities << endl;
    while(it <  maxIt && (1 - F) > acc) {
        if((1 - F)/acc > 100) eps = dc;
        else if((1 - F)/acc > 10) eps = 0.1*dc;
        else eps = 0.01*dc;
        #pragma omp parallel for
        for(int i = 0; i < Nmax; i++) {
            ArrayXf diff_vec(Nmax);
            diff_vec.setZero();
            for(int j = 0; j < Nmax; j++) if(i == j) diff_vec(j) = 1;
            getFidelity(x.head(Nmax) + eps*diff_vec, x.tail(Nmax), tp, dt, ket000, ket100, ket010, k000100, numFidelities, fidelities, dataList, checking_min);
            dFx(i) = fidelities(0) - F;
            getFidelity(x.head(Nmax), x.tail(Nmax) + eps*diff_vec, tp, dt, ket000, ket100, ket010, k000100, numFidelities, fidelities, dataList, checking_min);
            dFy(i) = fidelities(0) - F;
        }
        grad << dFx/eps, dFy/eps;
        p = -(B.inverse())*grad;
        x += eps*p;
        // x.head(Nmax) += dFx; x.tail(Nmax) += dFy;
        getFidelity(x.head(Nmax), x.tail(Nmax), tp, dt, ket000, ket100, ket010, k000100, numFidelities, fidelities, dataList, checking_min);
        F = fidelities(0);
        it++;
        /*

        *
    }
    cout << "=== FINAL FIDELITIES ===\n" << fidelities << endl;
    cout << "=== NUM OF ITERATIONS ===\n" << it << endl;
    return;
}
*/
void basic_funcs_vec::evolveState(float dt, int Ncycles, MatrixXcd& initial, MatrixXcd& target, int tp, int tf, ArrayXf& cx, ArrayXf& cy, float Ohm, bool flush, ArrayXXf& dataList, float F, MatrixXcd& finalState) {
    float collapse, Ohm1, Ohm2, Ohm3;
    int tmax, Nmax, dataIndex, tcycle, tcurrent, listLength;
    MatrixXcd currentState, H;
    Ohm1 = Ohm; Ohm2 = Ohm; Ohm3 = Ohm;
    Nmax = cx.size();
    ArrayXf c1(Nmax), c2(Nmax);
    currentState = initial; tcurrent = 0; dataIndex = 0;
    finalState = MatrixXcd::Zero(6,6);
    if(flush) listLength = (tp*ceil(Ncycles/2.0) + tf*floor(Ncycles/2.0))/dt + 1;
    else listLength = (tp*Ncycles)/dt;
    dataList.setZero(2, listLength);
    dataList(0, 0) = 0;
    dataList(1, 0) = (target*currentState.adjoint()).cwiseAbs().trace();
    for(int i = 0; i < Ncycles; i++) {
        if(flush) {
            if(i%2 == 0) {
                tcycle = tp; collapse = collapseOn; c1 = cx; c2 = cy;
            } else {
                tcycle = tf; collapse = collapseOff; c1.setZero(); c2.setZero();
            }
        } else {
            tcycle = tp; collapse = collapseOn; c1 = cx; c2 = cy;
        }
        tmax = ceil(tcycle/dt);
        for(int t = 1; t <= tmax; t++) {
            if(flush) H = HP + pulse(t*dt, tcycle, c1, Nmax)*HX1
                             + pulse(t*dt, tcycle, c2, Nmax)*HY1
                             + pulse(t*dt, tcycle, c1, Nmax)*HX2
                             + pulse(t*dt, tcycle, c2, Nmax)*HY2
                             + pulse(t*dt, tcycle, c1, Nmax)*HX3
                             + pulse(t*dt, tcycle, c2, Nmax)*HY3 + HS;
            else H = HP + Ohm1*HX1 + Ohm2*HX2 + Ohm3*HX3 + HS;
            dataList(0, dataIndex) = tcurrent + t*dt;
            dataList(1, dataIndex) = (target*currentState.adjoint()).cwiseAbs().trace();
            lindbladRK4(collapseOn, collapse, dt, H, currentState);
            dataIndex++;
        }
        tcurrent += tcycle;
    }
    int quarterPoint = dataList.cols()*0.25;
    F = dataList.block(0, 3*quarterPoint, 2, quarterPoint).rowwise().mean()(1);
    finalState = currentState;

    return;
}

