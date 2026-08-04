#ifndef GEOFIK_H  
#define GEOFIK_H

#include <array> 
#include <vector>
#include <eigen3/Eigen/Dense>
using namespace std;

constexpr double PI = 3.14159265359;

/**
 * @brief Computes the joint angles given a Jacobian and the rotation matrix of the ee frame.
 * @param J         transpose of J.
 * @param R         the rotation matrix of ee frame with respect to frame O. 
 * @param ee        [optional] name of ee frame ('E', 'F' or '8').
 * @return          joint angles q.
 */
array<double, 7> J_to_q(const array<array<double, 6>, 7>& J, const array<array<double, 3>, 3>& R, const char ee = 'E');

/**
 * @brief Computes the Jacobian given the joint angles.
 * @param q         joint angles, name of ee frame.
 * @param ee        [optional] Name of ee frame ('E', 'F', '8', ...,'1').
 * @return          transpose of J.
 */
array<array<double, 6>, 7> J_from_q(const array<double, 7>& q, const char ee = 'E');

/**
 * @brief Forward kinematics.
 * @param q         joint angles, 
 * @param ee        name of ee frame ('E', 'F', '8', ...,'1').
 * @return          transformation matrix of ee frame with respect to frame O.
 */
Eigen::Matrix4d franka_fk(const array<double, 7>& q, const char ee = 'E');

/**
 * @brief IK with q7 as free variable.
 * @param r         position of frame E with respect to frame O.
 * @param ROE       rotation matrix of frame E with respect to frame O (row-first format).
 * @param q7        joint angle of joint 7 (radians)
 * @param qsols     array to store 8 solutions
 * @param q1_sing   [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @return          number of solutions found.
 */
unsigned int franka_ik_q7(const array<double, 3>& r,
                          const array<double, 9>& ROE,
                          const double q7,
                          array<array<double, 7>, 8>& qsols,
                          const double q1_sing = PI / 2);

/**
 * @brief IK with q4 as free variable.
 * @param r         position of frame E with respect to frame O.
 * @param ROE       rotation matrix of frame E with respect to frame O (row-first format).
 * @param q4        joint angle of joint 4 (radians)
 * @param qsols     array to store 8 solutions
 * @param q1_sing   [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @param q7_sing   [optional] emergency value of q7 in case of singularity of S7 intersecting S (type-2 singularity).
 * @return          number of solutions found.
 */
unsigned int franka_ik_q4(const array<double, 3>& r,
                          const array<double, 9>& ROE,
                          const double q4,
                          array<array<double, 7>, 8>& qsols,
                          const double q1_sing = PI / 2,
                          const double q7_sing = 0);

/**
 * @brief IK with q6 as free variable.
 * @param r         position of frame E with respect to frame O.
 * @param ROE       rotation matrix of frame E with respect to frame O (row-first format).
 * @param q6        joint angle of joint 6 (radians)
 * @param qsols     array to store 8 solutions
 * @param q1_sing   [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @param q7_sing   [optional] emergency value of q7 in case of singularity of S7 intersecting S (type-2 singularity).
 * @return          number of solutions found.
 */
unsigned int franka_ik_q6(const array<double, 3>& r,
                          const array<double, 9>& ROE,
                          const double q6,
                          array<array<double, 7>, 8>& qsols,
                          const double q1_sing = PI / 2,
                          const double q7_sing = 0);

/**
 * @brief IK with swivel angle as free variable (numerical).
 * @param r         position of frame E with respect to frame O.
 * @param ROE       rotation matrix of frame E with respect to frame O (row-first format).
 * @param theta     swivel angle in radians (see paper for geometric defninition)
 * @param qsols     array to store 8 solutions
 * @param q1_sing   [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @param n_points  [optional] number of points to discretise the range of q7.
 * @return          number of solutions found.
 */
unsigned int franka_ik_swivel(const array<double, 3>& r,
                              const array<double, 9>& ROE,
                              const double theta,
                              array<array<double, 7>, 8>& qsols,
                              const double q1_sing = PI / 2,
                              const unsigned int n_points = 500,
                              const unsigned int n_fine_search = 3);

/**
 * @brief Calculates the swivel angle given the joint angles q.
 * @param q         joint angles.
 * @return          swivel angle theta (see paper for geometric defninition).
 */
double franka_swivel(const array<double, 7>& q);

/**
 * @brief IK to calculate Jacobian and joint angles with q7 as free variable.
 * @param r             position of frame E with respect to frame O.
 * @param ROE           rotation matrix of frame E with respect to frame O (row-first format).
 * @param q7            joint angle of joint 7 (radians).
 * @param Jsols         array to store 8 solutions for the Jacobians.
 * @param qsols         array to store 8 solutions for the joint angles.
 * @param joint_angles  [optional] if false only Jacobians are returned.
 * @param Jacobian_ee   [optional] ee frame of the Jacobian, not the IK ('E', 'F', '8' or '6').
 * @param q1_sing       [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @return              number of solutions found.
 */
unsigned int franka_J_ik_q7(const array<double, 3>& r,
                            const array<double, 9>& ROE,
                            const double q7,
                            array<array<array<double, 6>, 7>, 8>& Jsols,
                            array<array<double, 7>, 8>& qsols,
                            const bool joint_angles = false,
                            const char Jacobian_ee = 'E',
                            const double q1_sing = PI / 2);

/**
 * @brief IK to calculate Jacobian and joint angles with q4 as free variable.
 * @param r             position of frame E with respect to frame O.
 * @param ROE           rotation matrix of frame E with respect to frame O (row-first format).
 * @param q4            joint angle of joint 4 (radians).
 * @param Jsols         array to store 8 solutions for the Jacobians.
 * @param qsols         array to store 8 solutions for the joint angles.
 * @param joint_angles  [optional] if false only Jacobians are returned.
 * @param Jacobian_ee   [optional] ee frame of the Jacobian, not the IK ('E', 'F', '8' or '6').
 * @param q1_sing       [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @param q7_sing       [optional] emergency value of q7 in case of singularity of S7 intersecting S (type-2 singularity).
 * @return              number of solutions found.
 */
unsigned int franka_J_ik_q4(const array<double, 3>& r,
                            const array<double, 9>& ROE,
                            const double q4,
                            array<array<array<double, 6>, 7>, 8>& Jsols,
                            array<array<double, 7>, 8>& qsols,
                            const bool joint_angles = false,
                            const char Jacobian_ee = 'E',
                            const double q1_sing = PI / 2,
                            const double q7_sing = 0);

/**
 * @brief IK to calculate Jacobian and joint angles with q6 as free variable.
 * @param r             position of frame E with respect to frame O.
 * @param ROE           rotation matrix of frame E with respect to frame O (row-first format).
 * @param q6            joint angle of joint 6 (radians).
 * @param Jsols         array to store 8 solutions for the Jacobians.
 * @param qsols         array to store 8 solutions for the joint angles.
 * @param joint_angles  [optional] if false only Jacobians are returned.
 * @param Jacobian_ee   [optional] ee frame of the Jacobian, not the IK ('E', 'F', '8' or '6').
 * @param q1_sing       [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @param q7_sing       [optional] emergency value of q7 in case of singularity of S7 intersecting S (type-2 singularity).
 * @return              number of solutions found.
 */
unsigned int franka_J_ik_q6(const array<double, 3>& r,
                            const array<double, 9>& ROE,
                            const double q6,
                            array<array<array<double, 6>, 7>, 8>& Jsols,
                            array<array<double, 7>, 8>& qsols,
                            const bool joint_angles = false,
                            const char Jacobian_ee = 'E',
                            const double q1_sing = PI / 2,
                            const double q7_sing = 0);

/**
 * @brief IK to calculate Jacobian and joint angles with swivel angle as free variable (numerical).
 * @param r             position of frame E with respect to frame O.
 * @param ROE           rotation matrix of frame E with respect to frame O (row-first format).
 * @param theta         swivel angle in radians (see paper for geometric defninition).
 * @param Jsols         array to store 8 solutions for the Jacobians.
 * @param qsols         array to store 8 solutions for the joint angles.
 * @param joint_angles  [optional] if false only Jacobians are returned.
 * @param Jacobian_ee   [optional] ee frame of the Jacobian, not the IK ('E', 'F', '8' or '6').
 * @param q1_sing       [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @param n_points      [optional] number of points to discretise the range of q7.
 * @return              number of solutions found.
 */
unsigned int franka_J_ik_swivel(const array<double, 3>& r,
                                const array<double, 9>& ROE,
                                const double theta,
                                array<array<array<double, 6>, 7>, 8>& Jsols,
                                array<array<double, 7>, 8>& qsols,
                                const bool joint_angles = false,
                                const char Jacobian_ee = 'E',
                                const double q1_sing = PI / 2,
                                const unsigned int n_points = 600, 
                                const unsigned int n_fine_search = 3);

#endif