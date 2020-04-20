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
 *    AttrBinningEstimator.java
 *    Copyright (C) 2004
 *
 */
package weka.estimators;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.Instances;
import weka.core.Option;
import weka.core.Range;
import weka.core.Utils;
import weka.estimators.MultiBinningUtils.GlobalSplitData;
import weka.estimators.MultiBinningUtils.Split;
 
/**
 <!-- globalinfo-start -->
 * Abstract class of an estimator for one attribute that is used in a 
 * multi-dimensional estimator.
 <!-- globalinfo-leftEnd -->
 *
 * @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
 * @version $Revision: 1.0 $
 */
public abstract class AttrBinningEstimator extends AttrEstimator {
    
  /** illegal cut as it happens */
  public static int D_ILLCUT          = 5; // 6 
  
  /** trace through precedures */
  public static int D_ILLCUTCONTROL   = 15; // 16
  
  // stuff set by options
  /** range of outputtyp */
  protected Range m_checkLevels = new Range();
  
  /** some check levels are set */
  protected boolean m_checkFurther = false;
  
  /** Dont allow illegal cuts. */
  public static int C_CHECK_ILLEGAL_CUT = 1;
  protected boolean m_check_illegal_cut = true;
  
  /** list all cutpoints at leftEnd */
  public static int C_CHECK_TOOHIGH_EW10 = 2;
  protected boolean m_check_toohigh_ew10 = false;
  
  /** Dont allow empty areas. */
  public static int C_CHECK_EMPTY_AREA = 3;
  protected boolean m_check_empty_area = false;
  
  /** threshhold for some kinds of forbidden cuts */
  protected double m_TreshCutHeight = Double.MAX_VALUE;
  
  /** different splitting criteria */
  protected static final int STANDARD_SPLIT = 0;  
  protected static final int CV_SPLIT = 1;
  protected static final int FULL_SPLIT = 2;
  protected static final int WEIRD_SPLIT = 4;
  
  /** number of illegal cuts in final model */
  protected double m_numInstForIllCut = -1.0;
  
  //leftEnd- stuff for options
  
  /** cut points on this attribute */
  Vector m_splitList = new Vector();
    
  /** postfix for histo output  */
  protected String m_filePostfix = null;
  
  /** holds the choice of the splitting methods */
  protected int m_splitMethod = FULL_SPLIT;
  
  // different split states 
  /** split was found to be invalid */
  protected static final int SPLIT_INVALID = 0;  
  /** no split found */
  protected static final int SPLIT_NOTFOUND = 1;
  /** split checked and valid */
  protected static final int SPLIT_VALIDATED = 2;
  
  /**
   * initialize with new data set
   * @param data the dataset to initialize the attribute estimator
   * @param attrIndex the index of the attribute
   * @exception if initialize does not work 
   */
  public void initializeNewData(Instances data, int attrIndex, double min, double max) throws Exception{
    
    super.initializeNewData(data, attrIndex, min, max);
    defineForbiddenCut(data, attrIndex);
  }
  
  public void defineForbiddenCut(Instances inst, int attrIndex) throws Exception {
    //todo 
    if (m_check_toohigh_ew10) {
      //dbo.dpln("C_TOOHIGH_EW10");
      EqualWidthEstimator est = new EqualWidthEstimator();
      //Estimator.buildEstimator((Estimator) est, inst, attrIndex, 
      //    -1, -1, false);  
      dbo.dpln(D_ILLCUTCONTROL, est.toString());   
      double height = ((BinningEstimator)est).getMaxHeight();
      m_TreshCutHeight = height * 2.0;
      dbo.dpln(D_ILLCUTCONTROL, "Height-threshhold "+m_TreshCutHeight);
    } 
  }
  
  /**
   * 
   * @param leftNum
   * @param rightNum
   * @param totalN
   * @param leftDist
   * @param rightDist
   * @param totalL
   * @return
   */ 
  protected boolean splitNotForbidden(double leftNum, double rightNum, double totalN,
      double leftDist, double rightDist, double totalL) {
    
    // test if illegal cut
    if (m_check_illegal_cut) {
      if (MultiBinningUtils.isIllegalCut(leftNum, leftDist, totalL, totalN)) {
        if (dbo.dl(D_ILLCUT)) {
          double l = leftDist / totalL;
          dbo.dp(" Illegal cut LEFT:");
          dbo.dpln("\nl/l " + l +" len "+leftDist +" num "+ leftNum);
        }
        return false;
      }
      if (MultiBinningUtils.isIllegalCut(rightNum, rightDist, totalL, totalN)) { 
        if (dbo.dl(D_ILLCUT)) {
          double l = rightDist / totalL;
          dbo.dp(" Illegal cut RIGHT:");
          dbo.dpln("\nl/l " + l +" len "+rightDist +" num "+ rightNum);
        }
        return false;
      }
    }
    
    // test if other forbidden cuts
    if (m_checkFurther) {
      if (isForbiddenCut(leftNum, leftDist, totalL, totalN)
          || isForbiddenCut(rightNum, rightDist, totalL, totalN)) {
        return false;
      }
    }
    return true;
  }
  
  /**
   * Checks if it is a forbidden cut.
   *
   *@param numInst number of instances
   *@param width width of the bin
   *@param totalLen length of total range
   *@param totalNum number of all not missing instances
   */
  public boolean isForbiddenCut(double numInst, double width, 
      double totalLen,  double totalNum) {
    
    dbo.dpln(D_ILLCUTCONTROL,"check further");
    
    if (m_check_toohigh_ew10) {
      dbo.dpln(D_ILLCUTCONTROL,"check--too high after 2x 10EW");
      // density = the height of the bin
      double density = MultiBinningUtils.getDensity(numInst, width, totalNum);
      
      // compare height with threshhold
      if (density > m_TreshCutHeight) {
        dbo.dpln(D_ILLCUTCONTROL, "too high after 2x 10EW");
        return true;
      }
    }
    if (m_check_empty_area) {
      dbo.dpln(D_ILLCUTCONTROL, "check--empty cuts");
      if (numInst <= 0) {
        dbo.dpln(D_ILLCUTCONTROL, "empty cuts not allowed");
        return true;
      }
    }
    return false;
  }
  
  /**
   * Enters a value into the split list.
   * @param cutValue
   */
  public void setSplit(double cutValue) {
    
    Double newSplit = new Double(cutValue);
    for (int i = 0; i < m_splitList.size(); i++) {
      double value = ((Double)m_splitList.elementAt(i)).doubleValue();
      if (value > cutValue) {        
        m_splitList.add(i, newSplit);
        return;
      }   
    }
    m_splitList.add(newSplit);
  }
  
  /** 
   * Sets whether illegal cuts (<10% of length and <= 2 instances) should be
   * dissallowed
   * @param flag new flag value
   */
  public void setForbidIllegalCut(boolean flag) {
    m_check_illegal_cut = flag;
  }
  
  /** 
   * Sets whether illegal cuts are dissallowed
   * @return true if illegal cuts are set to be disallowed
   */
  public boolean getForbidIllegalCut() {
    return m_check_illegal_cut;
  }
  
  /**
   * Switches the outputs on that are requested from the option O
   * @param list list of integers, all are used for an output type
   */
  public void setCheckLevels(String list) {
    if (list.length() > 0) {
      m_checkFurther = true; 
      m_checkLevels.setRanges(list);
      m_checkLevels.setUpper(30);
    }
  }
  
  /**
   * Gets the current output type selection
   *
   * @return a string containing a comma separated list of ranges
   */
  public String getCheckLevels() {
    return m_checkLevels.getRanges();
  }
  
  /**
   * Sets the number of instances for illegal cut computation
   *
   * @param numInst  number of instances for illegal cut computation
   */
  public void setNumInstForIllCut(double numInst) {
    m_numInstForIllCut = numInst;
  }
  
  /**
   * Returns the number of instances for illegal cut computation
   *
   * @return max the maximum number of binss
   */
  public double getNumInstForIllCut() {
    return m_numInstForIllCut;
  }
  
  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -D  <br>
   * If set, estimator is run in debug mode and 
   * may output additional info to the console.<p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    
    super.setOptions(options);
    
    // output data 
    String outputRange = Utils.getOption('C', options);
    setCheckLevels(outputRange);
     
    // min number of instances in a cut otherwise it is an illegal cut
    String numInstForIllCut = Utils.getOption('H', options);
    if (numInstForIllCut.length() != 0) {
      setNumInstForIllCut((double)Integer.parseInt(numInstForIllCut));
    } 
  }
  
  /**
   * Gets the current settings of the Estimator.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {
    
    Vector result = new Vector();
    // options of the superclass
    //String [] superOptions = super.getOptions();
    //for (int i = 0; i < superOptions.length; i++) {
    //  result.add(superOptions[i]);
    //}
    
    // verbose levels
    String checkLevels = getCheckLevels();
    if (checkLevels.length() > 0) {
      result.add("-C");
      result.add("" + checkLevels);
    }
    
    // min number of instances before illegal
    if (getNumInstForIllCut() > -1.0) {
      result.add("-H");
      result.add("" + getNumInstForIllCut());
    }
    
    return (String[])result.toArray(new String[result.size()]);
  }
  
  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    
    Vector newVector = new Vector(4);
    newVector.addElement(
        new Option(
            "\tSwitch on check mode and give list of forbidden cut types.\n"
            + "\teg: 1,2,11",
            "C", 1, "-C <option list>"));
    newVector.addElement(
        
        new Option(
            "\tMinimal number of instances allowed in bin,\n"
            + "\totherwise cut is avoided.\n",
            "H", 1, "-H <num>"));
    
    //Enumeration enu = super.listOptions();
    //while (enu.hasMoreElements()) {
    //  newVector.addElement(enu.nextElement());
    //}
    return newVector.elements();
  }
  
  protected abstract Split findOneSplit(GlobalSplitData sp, MultiBin bin) throws Exception;
  
  /**
   * @param args
   */
  public static void main(String[] args) {
    // TODO Auto-generated method stub
    
  }
  
}
