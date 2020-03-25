/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    LoglikeliComputer.java
 *    Copyright (C) 2004 Gabi Schmidberger
 *
 */

package weka.estimators;

import weka.core.Instances;
import weka.core.Debug.DBO;
import weka.filters.unsupervised.attribute.Bin;


/** 
 <!-- globalinfo-start -->
 * Density estimator; the density is estimated by discretizing the attribute
 * using the TUBE algorithm.
 * TUBE builds a binary density estimation tree. It first selects an optimal
 * cut point in each range by maximizing the log-likelihood.
 * The number of splits is selected using crossvalidated log-likelihood.
 * The order of splits is given by taking as next split the one with highest
 * loglikelihood gain.
 <!-- globalinfo-leftEnd -->
 * 
  *
 * @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
 * @version $Revision: 1.0 $
 */
public class LoglikeliComputer {
  
  private double m_loglikelihood = Double.NaN;
  
  /** total number of instances */
  private int m_numInstances = -1;
  
  /**
   * Get the current cut value.
   * @param data
   * @param index
   * @param offset
   * @param middleFlag
   * @return 
   
  private double getDiffInBetween(Instances data, int attrIndex, 
      int offset, int divide) { 
    double diff = 0.0;
    if (offset < data.numInstances() - 1) {
      diff = data.instance(offset + 1).value(attrIndex) 
      - data.instance(offset).value(attrIndex);
      diff = (diff / (double)divide);
    }     
    return diff;
  } **/
  
  /*
   * Compute the criteria for one cutpoint. 
   *
   * @param num number of instances left off the cut point
   * @param length volume (length) of the area 
   * @param numRight number of instances in that area
   */
  protected  double getCriteriaFromBin(Bin bin) {
    double criteria = 0.0;
//  ***
//  if (getLeastSquaresCV()) {
//  criteria = bin.getSquaredLikelihood();
//  double criteria2 = bin.getLikelihood();
//  criteria = criteria - 2.0 * criteria2;
//  } else {
    try {
      //dbo.dpln("getCriteriaFromBin");
      criteria = bin.getWeightLoglikelihood();
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
    //   }
    return criteria;
  }
  
  /*
   * Compute the criteria for one cutpoint. 
   *
   * @param rightNum number of instances right to the cut point
   * @param rightWidth length of the right range 
   * @param leftNum number of instances left off the cut point
   * @param rightNum volume (length) of the area 
   * @param totalNum number of instances in the total range
   */
  protected  double getCutCriteria(double rightNum, double rightWidth, 
      double leftNum, double leftWidth, double totalNum) {
    double rCriteria = getLoglikelihood(rightNum, rightWidth, totalNum);
    double lCriteria = getLoglikelihood(leftNum, leftWidth, totalNum);
    double criteria = rCriteria + lCriteria;
    return criteria;
  }
  
  /*
   * Compute the criteria for one cutpoint. 
   *
   * @param num number of instances left off the cut point
   * @param length volume (length) of the area 
   * @param numRight number of instances in that area
   */
  protected  double getCutCriteria(double num, double width, double totalNum) {
    if (num == 0.0) return 0.0;
    double criteria = 0.0;
    criteria = getLoglikelihood(num, width, totalNum);
    return criteria;
  }
  
  /*
   * Compute the mise criteria for one cutpoint. 
   *
   * @param rightNum number of instances right to the cut point
   * @param rightWidth length of the right range 
   * @param leftNum number of instances left off the cut point
   * @param leftNum volume (length) of the area 
   * @param totalNum number of instances in the total range
   */
  protected  double getLeastSquaresCutCriteria(double rightNum, double rightWidth, 
      double leftNum, double leftWidth, 
      double totalNum) {
    double rCriteria = getLikelihood(rightNum, rightWidth, totalNum);
    double lCriteria = getLikelihood(leftNum, leftWidth, totalNum);
    double lk = ((rightNum * rCriteria)  + (leftNum * lCriteria)) / totalNum;
    
    double squared = getSquaredCutCriteria(rightNum, rightWidth, leftNum,  leftWidth, totalNum);
    double criteria = - squared + 2.0 * lk;
    return criteria;
  }
  
  /*
   * Compute the suared part of the least squares cut criteria. 
   *
   * @param rightNum number of instances right to the cut point
   * @param rightWidth length of the right range 
   * @param leftNum number of instances left off the cut point
   * @param leftNum volume (length) of the area 
   * @param totalNum number of instances in the total range
   */
  protected  double getSquaredCutCriteria(double rightNum, double rightWidth, 
      double leftNum, double leftWidth, 
      double totalNum) {
    double rSquared = getSquaredArea(rightNum, rightWidth, totalNum);
    double lSquared = getSquaredArea(leftNum, leftWidth, totalNum);
    double squared = rSquared + lSquared;
    return squared;
  }
  
  /*
   * Compute the squared likelihood for one cutoff area.
   * @param num number of instances left off the cut point
   * @param length volume (length) of the area 
   * @param numRight number of instances in that area
   */
  protected  double getSquaredLikelihood(double num, double width, double totalNum) {
    if (num == 0.0) return 0.0;
    double llh = num / (width * totalNum);
    return llh * llh;
  }
  
  /*
   * Compute the squared likelihood for one cutoff area.
   * @param num number of instances left off the cut point
   * @param length volume (length) of the area 
   * @param numRight number of instances in that area
   */
  protected  double getSquared(double num, double width, double totalNum) {
    if (num == 0.0) return 0.0;
    double llh =  num / totalNum;
    llh = llh * llh;
    llh =  llh / width;
    return llh;
  }
  
  /*
   * Compute the squared likelihood for one cutoff area.
   * @param num number of instances left off the cut point
   * @param length volume (length) of the area 
   * @param numRight number of instances in that area
   */
  protected  double oopsgetSquared(double num, double width, double totalNum) {
    if (num == 0.0) return 0.0;
    double llh =  num / totalNum;
    DBO.p(""+ num+ " "+width+" n/N "+llh);
    llh = llh * llh;
    DBO.p(" sq "+llh);
    llh =  llh / width;
    DBO.pln(" /width "+llh);
    return llh;
  }
  
  /*
   * Compute the density for one cutoff area. 
   *
   * @param num number of instances in the bin
   * @param width width of the bin 
   * @param totalNum total number of instances in that area
   */
  protected double getDensity(double num, double width, double totalNum) {
    if (num == 0.0) return 0.0;
    double llh = num / (width * totalNum);
    // double llh = num / (totalNum);
    return llh;
  }
  
  /*
   * Compute the likelihood for one cutoff area. 
   *
   * @param num number of instances left off the cut point
   * @param length volume (length) of the area 
   * @param numRight number of instances in that area
   */
  protected  double getLikelihood(double num, double width, double totalNum) {
    if (num == 0) return 0.0;
    double llh = num / (width * totalNum);
    return llh;
  }
  
  /**
   * Get area underneath the squared likelihood.
   * @param num number of instances in the sub range
   * @param width width of the sub range of the area 
   * @param totalNum number of instances in that area
   * @return the area underneath the squared likelihood 
   */
  public double getSquaredArea(double num, double width, double totalNum) {
    if (num == 0) return 0.0;
    double llh = num / (width * totalNum);
    double squared = llh * llh;
    squared = squared * width;
    //    Oops.pln("squared "+squared  );    
    return squared;
  }
  
  /*
   * Compute the loglikelihood for one cutoff area. 
   *
   * @param num number of instances left off the cut point
   * @param width width of the sub range of the area 
   * @param totalNum number of instances in that area
   */
  protected  double getLoglikelihood(double num, double width, double totalNum) {
    //dbo.dpln("getLoglikelihood num "+num+" width "+width+" totalNum "+totalNum);
    if (num == 0) return 0.0;
    double density = num / (width * totalNum);
    double llk  = (num) * (Math.log(density));
    
    //dbo.dpln("Loglikelihood "+llk);
    return llk;
  }
  
  /**
   * Compute the likelihood of a list of integervalues and lenght of bins, 
   * that represent the number
   * of instances in bins and the widths of the bins
   * @param argString string with the parameters of number of instances and widths
   */
  public double computeLikelihood(String argString) {
    
    String [] args = argString.split("[ \t]");
    double llk = computeLikelihood(args);
    return llk;
  }
  
  /**
   * Compute the likelihood of a list of integervalues and lenght of bins, 
   * that represent the number
   * of instances in bins and the widths of the bins
   * @param argv alternating number of instances and widths of the bins
   *
   */
  public double computeLikelihood(String [] argv) {

    int argLen = argv.length;
    int k = (int) ((double)argLen / 2.0);
    int [] nums = new int[k];
    double [] widths = new double[k];
    int j = 0;
    int N = 0;
    double llk = 0.0;
    for (int i = 0; i < k; i++, j++) {
      nums[i] = Integer.parseInt(argv[j]);
      N += nums[i];
      j++;
      widths[i] = Double.parseDouble(argv[j]);
      llk += getLoglikelihood(nums[i], widths[i], N);
    }
    m_loglikelihood = llk;
    m_numInstances = N;
    return llk;
  }
  
  public int getNumInst() {
    return m_numInstances;
  }
  
  /**
   * Display a representation of the computation.
   *@return a string with the results
   */
  public String toString() {

    String text = ("Number Instances "+m_numInstances+" Loglikelihood is "+m_loglikelihood);
    return text;
  }

  /**
   * Main method for testing this class.
   *
   * @param argv should contain the options for the estimator
   */
  public static void main(String [] argv) {
    
    try {
//    DBO.pln("argument 0 "+argv[0]);
       
      LoglikeliComputer comp = new LoglikeliComputer();
      comp.computeLikelihood(argv);
      System.out.println(comp.toString());
      
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
  }
}
