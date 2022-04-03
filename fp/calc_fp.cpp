#define _USE_MATH_DEFINES
#include <cppsim/state.hpp>
#include <Eigen/QR>
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <deque>
#include <chrono>
#include <iomanip>
#include <string>
#include <complex>
//#include <Eigen/KroneckerProduct>
#include <filesystem>
#include <cmath>

ComplexMatrix my_tensor_product(ComplexMatrix A, ComplexMatrix B) {
    /** Function to compute the tensor product (Kronecker product, np.kron(A,B))
     *   Args:
     *       A,B(ComplexMatrix) := Complex matrices
     *   Return:
     *       A tensor B (np.kron(A,B)) := Matrix of the result of the tensor product of A and B
     */
    //Get the dimension of A and B matrices
    unsigned long long int mat_A_dim = A.rows();
    unsigned long long int mat_B_dim = B.rows();
    //Declare a matrix to save the calculation results
    ComplexMatrix C(mat_A_dim*mat_B_dim, mat_A_dim*mat_B_dim);
    //Declare a matrix that represents some blocks of the matrix
    ComplexMatrix small_mat;

    //Calculation of tensor products
    for (int i = 0; i < mat_A_dim; i++) {
        for (int j = 0; j < mat_A_dim; j++) {
            //Calculate the scalar multiples of matrix B by the elements of matrix A to create a block matrix
            small_mat = A(i, j) * B;
            //Insert the block matrix value to the appropriate part of the matrix after the tensor product
            for (int k = 0; k < mat_B_dim; k++) {
                for (int l = 0; l < mat_B_dim; l++) {
                    C(k + i * mat_B_dim, l + j * mat_B_dim) = small_mat(k, l);
                }
            }
        }
    }
    return C;
}

/**
 * Ref: https://github.com/qulacs/qulacs/blob/master/src/cppsim/gate_factory.cpp#L180
 */
ComplexMatrix gen_haar_RU(unsigned int num_qubits) {
    /** Function to generate an n-qubit Haar Random Unitary.
     *  Implemented with reference to the one implemented in Qulacs.
     *  Arg:
     *      num_qubit(int) := number of qubit
     *  Return:
     *      Q(ComplexMatrix) := matrix of Haar Random Unitary
     */
    Random random;
    unsigned long long int dim = pow(2, num_qubits);
    ComplexMatrix matrix(dim, dim);
    for (unsigned long long int i = 0; i < dim; ++i) {
        for (unsigned long long int j = 0; j < dim; ++j) {
            matrix(i, j) = (random.normal() + 1.i * random.normal()) / sqrt(2.);
        }
    }
    Eigen::HouseholderQR<ComplexMatrix> qr_solver(matrix);
    ComplexMatrix Q = qr_solver.householderQ();
    // actual R matrix is upper-right triangle of matrixQR
    auto R = qr_solver.matrixQR();
    for (unsigned long long int i = 0; i < dim; ++i) {
        CPPCTYPE phase = R(i, i) / abs(R(i, i));
        for (unsigned long long int j = 0; j < dim; ++j) {
            Q(j, i) *= phase;
        }
    }
    return Q;
}

/** Function to return the current date and time as string
 * Used to specify initial seed values, file names, etc.
 * Example) 22:58:45, June 29, 2021  => "20210629225845"
 */
inline std::string getDatetimeStr() {
#if defined(__GNUC__) || defined(__INTEL_COMPILER)
    //Linux(GNU, intel Compiler)
    time_t t = time(nullptr);
    const tm* localTime = localtime(&t);
    std::stringstream s;
    s << "20" << localTime->tm_year - 100;
    //zerofill using setw() and setfill()
    s << std::setw(2) << std::setfill('0') << localTime->tm_mon + 1;
    s << std::setw(2) << std::setfill('0') << localTime->tm_mday;
    s << std::setw(2) << std::setfill('0') << localTime->tm_hour;
    s << std::setw(2) << std::setfill('0') << localTime->tm_min;
    s << std::setw(2) << std::setfill('0') << localTime->tm_sec;
#elif _MSC_VER
    //Windows(Visual C++ Compiler)
    time_t t;
    struct tm localTime;
    time(&t);
    localtime_s(&localTime, &t);
    std::stringstream s;
    s << "20" << localTime.tm_year - 100;
    //zerofill using setw() and setfill()
    s << std::setw(2) << std::setfill('0') << localTime.tm_mon + 1;
    s << std::setw(2) << std::setfill('0') << localTime.tm_mday;
    s << std::setw(2) << std::setfill('0') << localTime.tm_hour;
    s << std::setw(2) << std::setfill('0') << localTime.tm_min;
    s << std::setw(2) << std::setfill('0') << localTime.tm_sec;
#endif
    //return the value as std::string
    return s.str();
}

class FramePotential {
private:
    /* member variable */
    unsigned int num_qubits;    //number of qubit
    unsigned int depth;         //the circuit depth
    unsigned long long int dim; //dimention of a system
    unsigned int t;             //the order of t of t-design
    double epsilon;             //convergence error
    unsigned int patience;      //Number used to determine convergence

    std::string circuit;        //Specify "LRC" or "RDC" as string type
    unsigned int num_gates_depthOdd;  //Number of 2-qubit Haar random unitaries at odd order depths
    unsigned int num_gates_depthEven; //Number of 2-qubit Haar random unitaries at even order depths

    std::vector<double> result_oneshot;
    std::vector<double> result_mean;

    /* sample unitary randomly */
    ComplexMatrix sample_unitary();

    /* determine convergence */
    bool check_convergence();

public:
    /* constructor */
    FramePotential(std::string circ) {
        this->circuit = circ;
        this->num_qubits = 3;
        this->depth = 3;
        this->dim = 8;
        this->t = 3;
        this->num_gates_depthOdd = 1;
        this->num_gates_depthEven = 1;

        this->result_oneshot.clear();
        this->result_mean.clear();
    }

    /* setter of parameters */
    void set_paras(unsigned int Nq, unsigned int D, unsigned int dim_t, double eps, unsigned int pat);

    /* calculate the value of Frame Potential until it converges */
    void calculate();

    /* output the calculation result */
    void output();

    /* File output of calculation results */
    void save_result(std::string file_name);

    /* getter of calculation results */
    double get_result();
};

void FramePotential::set_paras(unsigned int Nq, unsigned int D, unsigned int dim_t, double eps = 0.0001, unsigned int pat = 5) {
    //Parameter setting from arguments
    this->num_qubits = Nq;
    this->dim = pow(2, Nq);
    this->depth = D;
    this->t = dim_t;
    this->epsilon = eps;
    this->patience = pat;

    //Preparation of "LRC"
    if (this->circuit == "LRC") {
        this->num_gates_depthOdd = Nq / 2;
        if (Nq % 2 == 1) {
            this->num_gates_depthEven = Nq / 2;
        }
        else {
            this->num_gates_depthEven = (Nq / 2) - 1;
        }
    //Assign 0 for the case other than "LRC"
    } else {
        this->num_gates_depthOdd = 0;
        this->num_gates_depthEven = 0;
    }
}

void FramePotential::calculate() {
    ComplexMatrix U, Vdag;
    unsigned long long int count = 1;

    this->result_oneshot.clear();
    this->result_mean.clear();

    while (true) {
        U = sample_unitary();
        Vdag = sample_unitary().adjoint();
        this->result_oneshot.emplace_back(pow(abs((U*Vdag).trace()), 2. * this->t));
        std::cout << "\r" << "  calculated " << count << " times..." << std::string(10, ' ');
        count++;
        if (check_convergence()) {
            break;
        }
    }
    std::cout << std::endl;

}

ComplexMatrix FramePotential::sample_unitary() {
    //the unitary matrix that will eventually be returned
    ComplexMatrix big_unitary = Eigen::MatrixXd::Identity(this->dim, this->dim);

    if (this->circuit == "LRC") {
        //Unitary matrix of one layer of LRC
        ComplexMatrix small_unitary;
        for (int i = this->depth; i > 0; i--) {
            //generate 2-qubit Haar random unitary and take tensor product
            if (i % 2 == 1) {
                //if the depth is odd
                small_unitary = gen_haar_RU(2);
                for (int j = 0; j < this->num_gates_depthOdd - 1; j++) {
                    small_unitary = my_tensor_product(small_unitary, gen_haar_RU(2));
                }
                //When both the number of qubits and the depth of the circuit are odd, the Identity is added at the end.
                if (this->num_qubits % 2 == 1) {
                    small_unitary = my_tensor_product(small_unitary, Eigen::MatrixXd::Identity(2, 2));
                }
            }
            else {
                //Whenever the circuit depth is even, the first is always Identity
                small_unitary = Eigen::MatrixXd::Identity(2, 2);
                for (int j = 0; j < this->num_gates_depthEven; j++) {
                    small_unitary = my_tensor_product(small_unitary, gen_haar_RU(2));
                }
                //If the number of qubits is even and the depth of the circuit is even, put Identity at the end.
                if (this->num_qubits % 2 == 0) {
                    small_unitary = my_tensor_product(small_unitary, Eigen::MatrixXd::Identity(2, 2));
                }
            }
            //Merging "num_qubits" size of unitaries created for each depth by applying
            big_unitary *= small_unitary;
        }
    }
    else if(this->circuit == "RDC") {
        //create 1-qubit Hadamard matrix
        ComplexMatrix hadmard_matrix_2d = Eigen::MatrixXd(2, 2);
        hadmard_matrix_2d(0, 0) = (1/sqrt(2)) * 1;
        hadmard_matrix_2d(0, 1) = (1/sqrt(2)) * 1;
        hadmard_matrix_2d(1, 0) = (1/sqrt(2)) * 1;
        hadmard_matrix_2d(1, 1) = (1/sqrt(2)) * -1;
        //create "num_qubits"-qubit Hadamard matrix
        ComplexMatrix hadmard_matrix = hadmard_matrix_2d;
        for(int i=0;i<this->num_qubits-1;i++) {
            hadmard_matrix = my_tensor_product(hadmard_matrix, hadmard_matrix_2d);
        }
        //Create a vector whose elements are the diagonal components of a Random Diagonal Unitary
        ComplexMatrix RDU = Eigen::MatrixXd::Identity(this->dim, this->dim);
        //create a RDC
        for(int i=0;i<this->depth;i++) {
            //Put values in the diagonal components of RDU
            for(int j=0;j<this->dim;j++) {
                Random random;
                RDU(j,j) = std::exp(std::complex<float>(0.0, random.uniform() * 2 * M_PI));
            }
            //Repeat alternately applying RDU and "num_qubits"-qubit Hadmard for depth times
            big_unitary *= hadmard_matrix * RDU;
        }
    }
    else {
        std::cerr << "CAUTION: The circuit '" << this->circuit << "' is not implemented. Identity will be returned." << std::endl;
    }

    return big_unitary;
}

bool FramePotential::check_convergence() {
    bool flag = true;
    unsigned long long int num_calclated = this->result_oneshot.size();

    if (num_calclated == 1) {
        this->result_mean.emplace_back(result_oneshot[0]);
        flag = false;
    }
    else if (num_calclated < this->patience) {
        this->result_mean.emplace_back(((num_calclated - 1)*this->result_mean.back() + result_oneshot.back()) / num_calclated);
        flag = false;
    }
    else {
        this->result_mean.emplace_back(((num_calclated - 1)*this->result_mean.back() + result_oneshot.back()) / num_calclated);
        for (int i = 0; i < this->patience; i++) {
            if (abs(this->result_mean.back() - this->result_mean[num_calclated - 1 - this->patience + i]) >= this->epsilon) {
                flag = false;
                break;
            }
        }
    }
    return flag;
}

void FramePotential::output() {
    std::cout << std::endl << "*** Result ***" << std::endl;
    std::cout << "num_qubits : " << this->num_qubits << std::endl;
    std::cout << "depth : " << this->depth << std::endl;
    std::cout << "t : " << this->t << std::endl;
    std::cout << "epsilon : " << this->epsilon << std::endl;
    std::cout << "patience : " << this->patience << std::endl;
    std::cout << "FramePotential : " << this->result_mean.back() << std::endl << std::endl;
}

void FramePotential::save_result(std::string file_name = getDatetimeStr() + ".csv") {
    std::string path = "./result/" + file_name;

    std::ofstream csv_file(path);
    csv_file << "circ:" << this->circuit << ",t:" << this->t << std::endl;
    csv_file << "Nq:" << this->num_qubits << ",depth:" << this->depth << std::endl;
    csv_file << "epsilon:" << this->epsilon << ",patience:" << this->patience << std::endl;
    csv_file << "shot,average" << std::endl;

    for (unsigned long long int i = 0; i < this->result_oneshot.size(); i++) {
        csv_file << this->result_oneshot[i] << "," << this->result_mean[i] << std::endl;
    }
    csv_file.close();
}

double FramePotential::get_result() {
    return this->result_mean.back();
}


int main() {
    //Execution example
    /* set parameters */
    int ntimes = 10;
    std::vector<int> Nq_list = { 7 };
    std::vector<int> depth_list = { 11,12,13,14,15 };
    std::vector<int> t_list = { 2,3,4,5 };
    double eps = 0.0001;
    unsigned int pat = 5;
    /* Specify the circuit */
    std::string circ_type = "LRC";
    //std::string circ_type = "RDC";
    
    /* call the class */
    FramePotential FP = { circ_type };
    
    /* begin calculation */
    for (int i = 0; i < t_list.size(); i++) {
        for (int j = 0; j < Nq_list.size(); j++) {
            for (int k = 0; k < depth_list.size(); k++) {
                //set parameters
                FP.set_paras(Nq_list[j], depth_list[k], t_list[i], eps, pat);
                std::cout << std::endl << "Now => Nq:" << Nq_list[j] << ", depth:" << depth_list[k] << ", t:" << t_list[i] << std::endl;
                //set directory name
                std::string dir_name = circ_type
                                        + "_Nq" + std::to_string(Nq_list[j])
                                        + "_depth" + std::to_string(depth_list[k])
                                        + "_t" + std::to_string(t_list[i]);
                //making directory
                std::filesystem::create_directory("result/" + dir_name);
                //Variables for calculating mean and standard deviation values
                double ave = 0.0, std = 0.0;
                //repeat "n" times
                for(int n=0; n < ntimes; n++) {
                    //begin calculating the value of Frame Potential
                    FP.calculate();
                    //FP.output();
                    //save the log
                    FP.save_result(dir_name + "/n=" + std::to_string(n) + ".csv");
                    //get the calculation results and add them up
                    ave += FP.get_result();
                    std += (FP.get_result() * FP.get_result());
                }
                //calculate the average
                ave /= ntimes;
                //calculate the standard deviation
                std = sqrt(std / ntimes - ave * ave);
                //output the result
                std::cout << "  result : " << ave << "±" << std << std::endl;
                //save the result
                std::ofstream result_file("./result/" + dir_name + "/ave_std.txt");
                result_file << ave << "±" << std << std::endl;
                result_file.close();
            }
        }
    }

    return 0;
}