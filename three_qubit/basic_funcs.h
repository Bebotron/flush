// Operator declarations

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>
// #include <Eigen>
// #include <KroneckerProduct>

using namespace std;
using namespace Eigen;

#define PI 3.14159265358979323846
#define IM std::complex<double> (0.0, 1.0)

#ifndef BASIC_FUNCS_H_VEC
#define BASIC_FUNCS_H_VEC

class basic_funcs_vec {
    public:
        // Default Constructor
        // basic_funcs_vec();

        // Overload Constructor
        basic_funcs_vec(float, float, float);

        // Destructor
        ~basic_funcs_vec();

        // System functions
        MatrixXcd tensor(MatrixXcd*, int);
        VectorXcd tensor_vec(VectorXcd*, int);

        inline float pulse(float, int, ArrayXf&, int);
        inline void timePropagator(float, VectorXcd&, MatrixXcd&, VectorXcd&);
        inline void timePropagatorRK4(float, MatrixXcd&, VectorXcd&);
        inline void lindbladME(float, float, MatrixXcd&, MatrixXcd&, MatrixXcd&);
        inline void lindbladRK4(float, float, float, MatrixXcd&, MatrixXcd&);
        void getFidelity(ArrayXf, ArrayXf, int, float, VectorXcd&, VectorXcd&, VectorXcd&, VectorXcd&, int, ArrayXf&, ArrayXXf&, bool);
        void optimizePulse(float, float, int, float, float, ArrayXf&, ArrayXf&, VectorXcd&, VectorXcd&, VectorXcd&, VectorXcd&, ArrayXf&, ArrayXXf&, int, bool);
        // void BFGSoptimize(float, float, int, float, float, ArrayXf&, ArrayXf&, VectorXcd&, VectorXcd&, VectorXcd&, VectorXcd&, ArrayXf&, ArrayXXf&, int, bool);
        void evolveState(float, int, MatrixXcd&, MatrixXcd&, int, int, ArrayXf&, ArrayXf&, float, bool, ArrayXXf&, float, MatrixXcd&);

        // Member Variables
        float collapseOn, collapseOff;
        
    private:
        // Member Variables
        Matrix2cd a, I, X, Y, Z;
        MatrixXcd X1, X2, X3, Y1, Y2, Y3, Z1, Z2, Z3,
                  X1S, X2S, X3S, Y1S, Y2S, Y3S, Z1S, Z2S, Z3S,
                  a1, a1d, a2, a2d, a3, a3d,
                  HP, HS, HX1, HX2, HX3, HY1, HY2, HY3,
                  Eye;
};

#endif // BASIC_FUNCS_H

inline float basic_funcs_vec::pulse(float t, int tp, ArrayXf& c, int Nmax) {
    float f = 0;
    for(int n = 1; n <= Nmax; n++) {
        f += c[n - 1]*sin(n*PI*t/tp);
    }
    return f;
}

inline void basic_funcs_vec::timePropagator(float dt, VectorXcd& psi, MatrixXcd&        H, VectorXcd& output) {
    output = -2.0*PI*IM*H*dt*psi;
    return;
}

inline void basic_funcs_vec::timePropagatorRK4(float step, MatrixXcd& H,        VectorXcd& psi) {
    VectorXcd t1, t2, t3, k1, k2, k3, k4;
    timePropagator(step, psi, H, k1);
    t1 = psi + 0.5*k1; timePropagator(step, t1, H, k2);
    t2 = psi + 0.5*k2; timePropagator(step, t2, H, k3);
    t3 = psi + k3; timePropagator(step, t3, H, k4);
    psi = psi + (k1 + 2*k2 + 2*k3 + k4)/6.0;
    return;
}

inline void basic_funcs_vec::lindbladME(float cp, float cs, MatrixXcd& rho, MatrixXcd& H, MatrixXcd& output) {
    output = 2.0*IM*PI*(rho*H - H*rho) 
            + cp*(X1*rho*X1 - rho) + cs*(a1*rho*a1d - 0.5*(a1d*a1*rho +        rho*a1d*a1))
            + cp*(X2*rho*X2 - rho) + cs*(a2*rho*a2d - 0.5*(a2d*a2*rho +        rho*a2d*a2))
            + cp*(X3*rho*X3 - rho) + cs*(a3*rho*a3d - 0.5*(a3d*a3*rho +        rho*a3d*a3));
    return;
}

inline void basic_funcs_vec::lindbladRK4(float col1, float col2, float step, MatrixXcd& H, MatrixXcd& rho) {
    MatrixXcd t1, t2, t3, k1, k2, k3, k4;
    lindbladME(col1, col2, rho, H, k1);
    t1 = rho + 0.5*step*k1; lindbladME(col1, col2, t1, H, k2);
    t2 = rho + 0.5*step*k2; lindbladME(col1, col2, t2, H, k3);
    t3 = rho + step*k3; lindbladME(col1, col2, t3, H, k4);
    rho = rho + step*(k1 + 2*k2 + 2*k3 + k4)/6.0;
    return;
}