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
 *    AttrTUBE.java
 *    Copyright (C) 2009 Gabi Schmidberger
 */
package weka.estimators;


import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Capabilities.Capability;
import weka.core.Debug.DBO;
import weka.estimators.MultiBinningUtils.GlobalSplitData;
import weka.estimators.MultiBinningUtils.Split;

/** 
<!-- globalinfo-start -->
 *
 * Abstract class for TUBE-type binning estimators.
 *
<!-- globalinfo-leftEnd -->
 * @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
 * @version $Revision: 1.0 $
 **/

public abstract class AttrTUBE extends AttrBinningEstimator {

  /** output cutpoint and entropy value there */
  public static int D_INFOVERBOSE     = 0; // 1 on the command line
  
  /** output before each split */
  protected static int D_FOLLOWSPLIT     = 1; // 2 
  
  /** illegal cut as it happens */
  protected static int D_ILLCUT          = 5; // 6 
  
  /** output about all split possibilities */
  protected static int D_ABOUTSPLIT      = 10; // 11 
  
  /** output 10 cutpoints beween instances */
  protected static int D_SPLITACC        = 11; // 12
  
  /** trace through precedures */
  protected static int D_TRACE           = 14; // 15
      
  /**
   * prepares an attribute for cutting
   * @param data data set with instances
   * @param attrIndex index of the attribute to be prepared
   */
  public AttrTUBE() {  
    super();
   }
  
  /**
   * initialize with new data set
   * @param data the dataset to initialize the attribute estimator
   * @param attrIndex the index of the attribute
   * @exception if initialize does not work 
   */
  public void initializeNewData(Instances data, int attrIndex, double min, double max) throws Exception{   
    super.initializeNewData (data, attrIndex, min, max);
    }
  
  /**
   * finds one split
   * @param spInfo global data used for the split
   * @param bin the bin to find the split in
   * @param data the data set 
   * @param attrIndex the attributes index
   * @return the details of the split
   */
  public Split findOneSplit (GlobalSplitData spInfo, MultiBin bin) throws Exception {
    
    Split split = new Split();
    split.splitNumber = spInfo.splitCounter;
    split.attrIndex = m_attrIndex;
    split.trainCriterionDiff = 0.0;
    split.bin = bin;
    split = findMinInRange(spInfo, split);
    if (split == null) return null;
    
    // no split found, reason no cut with better criterion found
    if (split.trainCriterionDiff < 0.0) {
      // entropy didn't decrease
      spInfo.numTotallyUniform++;
      
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      dbo.dpln(D_SPLITACC, "Split not accepted : no minimum found ["
          +bin.getMinValue(m_attrIndex)+":"+bin.getMaxValue(m_attrIndex)+"]");
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --       
      return null;
    }
    
    // new min found    
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    // output for each new split found (split not yet executed!)
    if (dbo.dl(D_FOLLOWSPLIT)) {
      dbo.dpln("#new split found-" + split.attrIndex + "-- " + split.cutValue + 
          " minLeftNum " + split.leftNum + 
          " minRightNum " + split.rightNum + " minIndex " + split.index);
    }
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    
    if (splitAccepted(spInfo, split)) {
      // new splitpoint accepted
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      if (dbo.dl(D_SPLITACC)) {
        dbo.dpln("#Split accepted "+split.cutValue);
      }
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      return split;
    } else {
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      // reason no split found = split accepted failed
      if (dbo.dl(D_SPLITACC)) {
        dbo.dpln("#Split not accepted "+split.cutValue);
      }
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      return null;
    }
  }
  
  /**
   * Tests if split can be accepted
   * 
   * @param sp global splitting infos
   * @param split split information
   * @return true if split is accepted
   */ 
  protected boolean splitAccepted(GlobalSplitData sp, 
      Split split) {
    //dbo.dpln("splitAccepted: oldEntropy "+oldEntropy+
    //      " splitEntropy "+splitEntropy);
    double sum = split.leftNum + split.rightNum;
    double penalty = 0.0;
    
    /** output difference of entropys, before/after split
     double diff = (oldEntropy - splitEntropy);
     dbo.dpln(" || DIFF "+ diff+" penalty "+penalty);*/
    
    //dbo.dpln("m_splitMethod "+m_splitMethod);
    switch (m_splitMethod) {
    case STANDARD_SPLIT:
      //dbo.dpln("STANDARD_SPLIT");
      penalty = Math.log(sp.bigN) + Math.log(2.0);
      break;
    case CV_SPLIT:
      //dbo.dpln("CV_SPLIT");
      // decision is made before the split is selected
      return true;
    case FULL_SPLIT:
      //dbo.dpln("FULL_SPLIT");
      penalty = 0.0;
      break;
      
    case WEIRD_SPLIT:
      //dbo.dpln("WEIRD_SPLIT");
      // todo + instead of minus
      penalty =  - (sp.bigN * (Math.log(sum / sp.bigN))) / 10.0;
      break;
    }// switch
    
    
    if (split.oldCriterion < (split.newCriterion - penalty)) {
      // make the split
      return true;
    }
    
    if (dbo.dl(D_FOLLOWSPLIT)) {
      dbo.dpln("#criterion didn't increase ");
      dbo.dpln("#penalty "+penalty+" oldCriterion "+split.oldCriterion+" afterSplitCriterion "+split.newCriterion);
    }
    
    // reject the split
    return false;
  }
   
  /**
   * Find the cut point in the range with min criteria
   * @param sp the global data 
   * @param split the split specific data
   * @return the split data
   */
  protected abstract Split findMinInRange(GlobalSplitData sp, Split split) throws Exception;
    
  /**
   * Test if value is valid
   * @param ii
   * @param binValid
   * @return
   */
  protected boolean isValid(int ii, boolean[] valid, boolean[] binValid) {
    int aIndex = 0;
    int iIndex = 1;
    //DBO.pln(" "+ii);
    int index = (int)m_data.instance(ii).value(iIndex);
    boolean v = (valid[ii] && binValid[index]);
    return v;
  }
  
  /**
   * Give the number of valid values
   * @param leftBegin
   * @param leftEnd
   * @param binValid
   * @return number of valid found 
   */
  protected int getNumValid(int begin, int end, boolean[] valid, boolean[] binValid) {
    int num = 0;
    DBO.pln("leftBegin "+begin+" leftEnd "+end);
    for (int ii = begin; ii <= end; ii++) {
      if (isValid(ii, valid, binValid)) {
        num++;
      }
    }
    return num;
  }
  
  /**
   * returns the given index or the next valid index
   * @param oldOffset
   * @param valid
   * @param binValid
   * @return
   */
  protected int getNextLargerEqualValidIndex(int oldOffset, boolean[] valid, boolean[] binValid) {
    int ii = oldOffset;
      if (ii >= valid.length) return ii;
      while (!isValid(ii, valid, binValid)) {
        ii++;
        if (ii >= valid.length) return ii;
      }
      return ii;
    }
   
 /**
   * returns the given index or the next smaller valid index
   * @param oldOffset
   * @param valid
   * @param binValid
   * @return
   */
    protected int getNextSmallerEqualValidIndex(int oldOffset, boolean[] valid, boolean[] binValid) {
      int ii = oldOffset;
      if (ii < 0) return -1;
      while (!isValid(ii, valid, binValid)) {
        ii--;
        if (ii < 0) return ii;
      }
      return ii;
    }
    
    /**
     * Get the first offset, and if flag equal is true it could be the one exactly on the min.
     * @param cutValue current cut value
     * @param equal if equal is allowed
     * @param oldOffset the old offset
     * @param maxValue the largest value in the range
     * @param binValid flags if instance is valid in this bin
     * @return the first offset of a valid value in the range
     */
    protected int findFirstOffset(double cutValue, boolean equal, int oldOffset, double maxValue, 
        boolean [] binValid) { 
      
      int numInst = m_data.numInstances();
      int ii = oldOffset;
      
      double value = Double.NaN;
      while (Double.isNaN(value) && ii >= 0) {
        if (ii >= numInst) { 
          ii = -1;
          break;
        }
        else {
          if (isValid(ii, m_valid, binValid)) {
            double v = getValue(ii);
            if (v > maxValue) { 
              ii = -1;
              break;
            }
            else {
              if (equal) {
                if (v >= cutValue) value = v;           
              } else {
                if (v > cutValue) value = v;
              }
            }
          }
        }
        if (Double.isNaN(value)) ii++;
      }
      return ii;   
    }
      
    /**
     * Get the last offset, and if flag equal is true it could be the one exactly on the max.
     * @param equal if equal is allowed
     * @param oldOffset offset of instance smaller than maxValue
     * @param maxValue the largest value in the range
     * @param binValid flags if instance is valid in this bin
     * @return the first offset of a valid value in the range and the num off instances found
     */
    protected int [] findLastOffset(boolean equal, int oldOffset, double maxValue, 
        boolean [] binValid) { 
      double value;
      int numInst = m_data.numInstances();
      int ii = oldOffset;
      int last = ii;
      boolean notFound = true;
      int num = 0;
      while (notFound && ii >= 0) {
        ii++;
        if (ii >= numInst) ii = -1;
        else {
          if (isValid(ii, m_valid, binValid)) {
            value = getValue(ii);
            if (equal) {
              if (value >= maxValue) notFound = false;           
            } else {
              if (value > maxValue) notFound = false;
            }
            if (notFound) {
              last = ii;
              num++;
            }
          }
        }
        if (notFound) last = ii;
      }
      int [] r = new int[2];
      r[0] = last; r[1] = num;
      return r;   
    }
      
  /*
   * Compute the criteria for one cutpoint. 
   *
   * @param bin the bin to get the criteria from  
   * @return the computed criteria
   */
  protected  double getCriteriaFromBin(MultiBin bin) {
    double criteria = 0.0;
    try {
      //dbo.dpln("getCriteriaFromBin");
      criteria = bin.getAttrLoglk(m_attrIndex);
      
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
    double rCriteria = MultiBinningUtils.getLoglikelihood(rightNum, rightWidth, totalNum);
    double lCriteria = MultiBinningUtils.getLoglikelihood(leftNum, leftWidth, totalNum);
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
    criteria = MultiBinningUtils.getLoglikelihood(num, width, totalNum);
    return criteria;
  }
  
   /**
   * Parses a given list of options. <p>
   * If set, estimator is run in debug mode and 
   * may output additional info to the console.<p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    
    super.setOptions(options);
    
     }
  
  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    
    // attributes
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.NOMINAL_ATTRIBUTES);
   return result;
  }

 /**
   * Get a probability estimate for a value.
   *
   * @param data the value to estimate the probability of
   * @return the estimated probability of the supplied value
   */
  public double getProbability(double data) {
    
    return 0.0;
  }
  
  /**
   * @param args
   */
  public static void main(String[] args) {
    // TODO Auto-generated method stub
    
  }
  
}
