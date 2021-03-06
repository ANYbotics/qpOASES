/*
 *	This file is part of qpOASES.
 *
 *	qpOASES -- An Implementation of the Online Active Set Strategy.
 *	Copyright (C) 2007-2017 by Hans Joachim Ferreau, Andreas Potschka,
 *	Christian Kirches et al. All rights reserved.
 *
 *	qpOASES is free software; you can redistribute it and/or
 *	modify it under the terms of the GNU Lesser General Public
 *	License as published by the Free Software Foundation; either
 *	version 2.1 of the License, or (at your option) any later version.
 *
 *	qpOASES is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *	See the GNU Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public
 *	License along with qpOASES; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

/**
 *	\file test/cpp/TestExample4.cpp
 *	\author Hans Joachim Ferreau
 *	\version 3.2
 *	\date 2007-2017
 *
 *	Very simple example for testing qpOASES (using the possibility to specify user-defined constraint product function).
 */

#include <gtest/gtest.h>

#include "qp_oases/qpOASES.hpp"

//! Example for qpOASES main function using the possibility to specify user-defined constraint product function.
TEST(TestExamle4, SQProblemWithUserDefinedConstraintProductFunction) {  // NOLINT
  USING_NAMESPACE_QPOASES

  /* Setup data of first QP. */
  real_t H[2 * 2] = {1.0, 0.0, 0.0, 0.5};
  real_t A[1 * 2] = {1.0, 1.0};
  real_t g[2] = {1.5, 1.0};
  real_t lb[2] = {0.5, -2.0};
  real_t ub[2] = {5.0, 2.0};
  real_t lbA[1] = {-1.0};
  real_t ubA[1] = {2.0};

  /* Setup data of second QP. */
  real_t H_new[2 * 2] = {1.0, 0.5, 0.5, 0.5};
  real_t A_new[1 * 2] = {1.0, 5.0};
  real_t g_new[2] = {1.0, 1.5};
  real_t lb_new[2] = {0.0, -1.0};
  real_t ub_new[2] = {5.0, -0.5};
  real_t lbA_new[1] = {-2.0};
  real_t ubA_new[1] = {1.0};

  /* Setting up SQProblem object. */
  SQProblem example(2, 1);

  /* Solve first QP. */
  int_t nWSR = 10;
  example.init(H, g, A, lb, ub, lbA, ubA, nWSR, 0);

  real_t xOpt[2];
  real_t yOpt[2 + 1];
  example.getPrimalSolution(xOpt);
  example.getDualSolution(yOpt);

  /* Compute KKT tolerances */
  real_t stat, feas, cmpl;
  SolutionAnalysis analyzer;

  analyzer.getKktViolation(&example, &stat, &feas, &cmpl);
  printf("stat = %e\nfeas = %e\ncmpl = %e\n", stat, feas, cmpl);

  ASSERT_LT(stat, 1e-15);
  ASSERT_LT(feas, 1e-15);
  ASSERT_LT(cmpl, 1e-15);

  /* Solve second QP. */
  nWSR = 10;
  example.hotstart(H_new, g_new, A_new, lb_new, ub_new, lbA_new, ubA_new, nWSR, 0);

  /* Get and print solution of second QP. */
  example.getPrimalSolution(xOpt);
  example.getDualSolution(yOpt);
  printf("\nxOpt = [ %e, %e ];  objVal = %e\n\n", xOpt[0], xOpt[1], example.getObjVal());

  /* Compute KKT tolerances */
  analyzer.getKktViolation(&example, &stat, &feas, &cmpl);
  printf("stat = %e\nfeas = %e\ncmpl = %e\n", stat, feas, cmpl);

  ASSERT_LT(stat, 2e-15);
  ASSERT_LT(feas, 1e-15);
  ASSERT_LT(cmpl, 1e-15);
}