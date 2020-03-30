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
	 *    AttrTUBEPure.java
	 *    Copyright (C) 2009 Gabi Schmidberger
	 */
package weka.estimators;


import java.io.Serializable;

import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.Debug.DBO;
import weka.estimators.MultiBinningUtils.GlobalSplitData;
import weka.estimators.MultiBinningUtils.Split;
 
/** 
<!-- globalinfo-start -->
*
* Abstract class for TUBE-type pure binning estimators.
*
<!-- globalinfo-leftEnd -->
* @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
* @version $Revision: 1.0 $
**/
public final class AttrTUBEPure extends AttrTUBE {
  
 
/**
	 * 
	 */
	private static final long serialVersionUID = -6536171027446288234L;

private class AtomInfo implements Serializable {
    
    /**
	 * 
	 */
	private static final long serialVersionUID = -3606480925584522759L;
	int begin = -1;
    int end = -1;
    int num = -1;
    int nextBegin = -1;
    double value = Double.NaN;
    double nextValue = Double.NaN;
     
    boolean notValid() {
      return (Double.isNaN(value));
    }
  }

  /** the epsilon taken to cut beside a value  */
  private double m_epsilon = 1.0E-4;

  /**
   * Constructor  
   */
  public AttrTUBEPure() {
    
    super();
    
    // file postfix for histogram files and similar
    m_filePostfix = "TUp";  
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
    result.enable(Capability.NOMINAL_CLASS);
    return result;
  }

  /**
   * Gets the flag if
   * @return the setting of the epsilon cutting
   */
  public double getEpsilon() {
    return m_epsilon;
  }
  
  /**
   * Sets the epsilon value for epsiloncutting
   * @param eps the new value for epsilon
   */
  public void setEpsilon(double eps) {
    m_epsilon = eps;
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
   * Find the cut point in the range with min criteria
   * @param sp global data relevant for the split
   * @param split information about the split
   * @return the split performed
   */
  protected Split findMinInRange(GlobalSplitData sp, Split split) throws Exception {  
    //DBO.pln("findMinInRange \n "+m_data+"\nfindMinInRange");
    double minValue = split.bin.getMinValue(m_attrIndex);
    double maxValue = split.bin.getMaxValue(m_attrIndex);         
    //DBO.pln("attr "+m_attrIndex+" min "+minValue+"--"+maxValue);
    if (dbo.dl(D_ABOUTSPLIT)) {
      DBO.pln("find min in range " + minValue + "--" + maxValue);
    }
    if (minValue > maxValue) return null;
    
    int numInst = (int) split.bin.getWeight();
    // don't split empty bin
    if (numInst == 0) return null;
    
    int totalN = (int)sp.bigN;
    double totalL = sp.bigL[m_attrIndex];
    boolean [] binValid = split.bin.getValid();
    
    // sort out criteria
    double newCriterion = -Double.MAX_VALUE;
    double leftCriterion = 0.0;
    double rightCriterion = 0.0;
    
    // prepare split object
    split.oldCriterion = getCriteriaFromBin(split.bin);
    split.newCriterion = -Double.MAX_VALUE;
    
    //int leftBegin = split.bin.getBegin(split.m_attrIndex);
    int begin = 0;
    //DBO.pln(""+split.bin.fullResultsToString()+" "+m_attrIndex);
    if (dbo.dl(D_ABOUTSPLIT)) {
      DBO.pln(""+split.bin.toString()+" "+m_attrIndex);
      DBO.pln(""+split.bin.fullResultsToString()+" "+m_attrIndex);
    }
    AtomInfo atom = getNextAtom(begin, numInst, maxValue, binValid);
    if (atom.notValid()) {
      DBO.pln(""+split.bin.fullResultsToString()+" "+m_attrIndex);
      throw new Exception("TUBEPure: Find cut value failed");
    }
    int offset = atom.begin;
    
    // prepare cutvalue for while
    double cutEpsilon = m_epsilon;
    double cutValue = 0.0;   
    double bestCutValue = 0.0;
    
    double lastValue = minValue;
    int leftNum = 0;
    int rightNum = numInst;
    double leftDist = 0.0;
    double rightDist = maxValue - minValue;
    
    // if true, cut left of instance
    boolean leftToggle = false;
    
    // **************************************************************************
    // check over all instances starting from left
    while (atom != null) {
      
      leftToggle = !leftToggle;
      cutValue = getNextCutValue(atom.value, leftToggle, cutEpsilon);
      //DBO.pln("atom.value "+atom.value+" cutValue "+cutValue);
      boolean cutOk = false;
      double nextValue = atom.nextValue; 
      if (Double.isNaN(nextValue)) nextValue = maxValue;
      if (cutValue > lastValue && cutValue < nextValue) {
        cutOk = true;
      }
      //if (!cutOk) DBO.pln("cutnotok");
 
      // give more instances to the right
      if (!leftToggle) {
        rightNum -= atom.num;
        leftNum += atom.num;
        offset = atom.nextBegin;
      }
      //DBO.pln("toggle "+leftToggle+" right "+rightNum+" left "+leftNum+" offset "+offset+" value "+atom.value);
      if (cutOk) {
        leftDist = cutValue - minValue;
        rightDist = maxValue - cutValue; 
        
         newCriterion = getCutCriteria(leftNum, leftDist, rightNum, rightDist, totalN);
        
        // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
        // output all LLKs
        if (dbo.dl(D_ABOUTSPLIT)) {
          rightCriterion = getCutCriteria(rightNum, rightDist, totalN);
          leftCriterion = getCutCriteria(leftNum, leftDist, totalN);
          dbo.dpln(""+cutValue + " "+ newCriterion +" "+leftCriterion+" "+rightCriterion+
              " left "+leftNum+"/"+leftDist+" right "+rightNum+"/"+rightDist+
              " cut# "+cutValue);
        }
        // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
        
        // see if a new maximum
        if (newCriterion > split.newCriterion) {  
          // make checks
          if (splitNotForbidden(leftNum, rightNum, totalN,
              leftDist, rightDist, totalL)) {
            // current cut is new minimum 
            split.newCriterion = newCriterion;
            split.leftCriterion = leftCriterion;
            split.rightCriterion = rightCriterion;
            split.trainCriterionDiff = (split.newCriterion - split.oldCriterion);
            split.index = -1;
            split.lastLeft = atom.end;
            split.firstRight = atom.nextBegin;
            split.cutValue = cutValue;
            bestCutValue = cutValue;
            split.leftDist = leftDist;
            split.rightDist = rightDist;
            split.leftNum = leftNum;
            split.rightNum = rightNum;
            
          }
        }
      }    
      if (!leftToggle) {
        // left again so go to next
        if (atom.nextBegin < 0) atom = null;
        else {
          lastValue = atom.value;
          atom =  getNextAtom(atom.nextBegin, numInst, maxValue, binValid);
        }
      }
    } // leftEnd while
      
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    // output where split was found
    if (dbo.dl(D_ABOUTSPLIT)) {
      dbo.dpln("# best criterion found at "+split.leftNum+" "+split.newCriterion+" cut@ "+split.cutValue +
          " minValue "+minValue+" maxValue "+maxValue);
    }
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    return split;  
  }

  /**
   * Analyze the nextAtom. 
   * Get:
   * leftBegin offset,
   * num of Atoms, number in each atom
   * @param numInst number of values
   * @param minValue the smallestest value in the range
   * @param maxValue the largest value in the range
   * @return the first offset in the range
   */
  protected AtomInfo getNextAtom(int begin, int numInst, double maxValue, boolean [] binValid) {
    //DBO.pln("getNextAtom "+leftBegin+" "+numInst);
    AtomInfo atom = new AtomInfo();
    
    if (numInst <= 0) { 
      return atom;
    }
    // find first valid
    if ((begin == 0) && !isValid(begin, m_valid, binValid)) begin++;
    if (begin >= binValid.length) {
      // no instances found
      atom.begin = -1; 
      return atom; 
    }
    while (!isValid(begin, m_valid, binValid) && begin > 0) {
      begin++; 
      if (begin >= binValid.length) {
        // no instances found
        atom.begin = -1; 
        return atom; 
        }
    }
    atom.begin = begin;
    atom.end = begin;
    atom.value = getValue(begin);
    atom.nextValue = atom.value;
    atom.num = 1;
    int ii = atom.begin;
    while (atom.value == atom.nextValue) {
      ii++;
      if (ii >= binValid.length) {
        atom.nextValue = Double.NaN;
        atom.nextBegin = -1;
      }
      else {
        if (isValid(ii, m_valid, binValid)) {
          atom.nextValue = getValue(ii);
          if (atom.nextValue == atom.value) { 
            atom.end = ii;
            atom.num++;
          }
          if (atom.nextValue > maxValue) atom.nextValue = Double.NaN;
          else {
            if (atom.nextValue != atom.value) {
              atom.nextBegin = ii;
            }
          }
        }
      }
    }
    
    return atom;
  }
  
   /**
   * Get the next cut value, for the gridcut method.
   * @param oldCut the old cut value
   * @return 
   */
  private double getNextCutValue(double value, boolean leftToggle, double cutDist) {
    double cutValue = Double.NaN;
    
     if (leftToggle) {
      value -= cutDist;
    } else {
      value += cutDist;
    }
    return value;
  }
    
  /**
   * Get the first offset, could be the one exactly on the min.
   * @param cutValue current cut value
   * @param firstOffset the old offset
   * @param maxValue the largest value in the range
   * @return the first offset in the range
   */
  protected int findLastOffset(int prevOffset, double maxValue,
      boolean [] binValid) { 
    
    // attribute is always the first 
    int aIndex = 0;
    
    int ii = prevOffset;
    int last = ii;
    double value;
    int numInst = m_data.numInstances();
    while (ii < numInst) {
      last = ii;
      value = m_data.instance(ii).value(aIndex);
      if (value > maxValue) {
        if (last == prevOffset) return -1;
        return last;
        }
      if (value == maxValue) return ii;
      ii = getNextLargerEqualValidIndex(++ii, m_valid, binValid);
      if (ii == numInst) {
        if (last == prevOffset) return -1;
        return last;
      }
    }  
    
    return ii--;
  }  
 
 /**
   * Get the first instance offset with a value larger than the cut value.
   * @param cutValue current cut value
   * @param leftToggle if the cut is left of the point given by oldOffset
   * @param oldOffset the old offset
   * @return the offset of the first instance right of the cut, or
   * one after the last one
   */
  private int findOffsetAfterCut(double cutValue, boolean leftToggle, int oldOffset, 
      boolean [] binValid) { 
    if (oldOffset < 0) return -1;
 
    // attribute is always the first 
    int numInst = m_data.numInstances();
    int aIndex = 0;
  
    int ii = oldOffset;
    double value;
    boolean found = false;
    if (leftToggle) {
       // endless loop - not really 
      do {
        ii = getNextSmallerEqualValidIndex(ii, m_valid, binValid);
        if (ii < 0) found = true; 
        else {
          value = m_data.instance(ii).value(aIndex);
          if (value < cutValue) { found = true; }
          else { ii--; }
        }
      } while (!found) ;
    }
    if (ii < 0) ii = 0;
    found = false;
    // endless loop - not really 
    do {
      ii = getNextLargerEqualValidIndex(ii, m_valid, binValid);
      if (ii >= numInst) return -1;
      value = m_data.instance(ii).value(aIndex);
      if (value > cutValue) found = true;
      else {
        ii++;
      }
    } while (!found);
    
    return ii;
  }
    
  /**
   * Get the next valid offset in this bin or -1 if no more found.
   * @param oldOffset the old offset
   * @param binValid the valid list for this bin
   * @return the next valid offset
   * one after the last one
   */
  private int findNextOffset(int oldOffset, boolean [] binValid) { 
    int ii = getNextLargerEqualValidIndex(oldOffset + 1, m_valid, binValid);
    return ii;
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
    String epsilonString = Utils.getOption('Z', options);
    if (epsilonString.length() > 0) {
      setEpsilon(Double.parseDouble(epsilonString));
    }    
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
