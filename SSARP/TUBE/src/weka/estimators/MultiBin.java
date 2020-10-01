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
 *    MultiBin.java
 *    Copyright (C) 2009 Gabi Schmidberger
 *
 */

package weka.estimators;

import java.io.Serializable;
import java.util.Random;
import java.util.Vector;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.Debug.DBO;
import weka.estimators.MultiBinningUtils.GlobalSplitData;
import weka.estimators.MultiBinningUtils.Split;
import weka.estimators.MultiBinningUtils.Tree;

/** 
 *
 <!-- globalinfo-start -->
 * Class represents a multidimensional bin. Bin is produced by a discretizing algorithm.
 <!-- globalinfo-leftEnd -->
 *
 * @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
 * @version $Revision: 1.0 $
 */
public class MultiBin implements Comparable<MultiBin>, Serializable{

  /** different comparison criteria */
  /** use density */
  private static final int DENSITY = 1;
  
  /** use diffenece between dens of a and b instances */
  private static final int DIVDENSITY = 2;
   
  /** holds the choice of the comparison method */
  private int m_compareMethod = DENSITY;
 
  /** original data set */
  public Instances m_globalData = null; 
  
  /** data used (split up into one dataset for each attribute) */
  protected Instances [] m_data = null;
  
  /** attribute index of the last cut */
  protected int m_lastCutAttr = -1;
  
  /** split depth from which the bin originates from */
  protected int m_splitDepth;
  
  /** number of attributes in the dataset */
  protected int m_numAttr = -1;
  
  /** flags set if attributes are in use (not class etc.) */
  protected boolean [] m_usedAttr;
    
  /** contains flag if instance is valid for that attribute (index) or not */
   protected boolean [] m_valid;

  /** flag, bin that contains a difference */
  protected boolean m_isDifferenceBin = false;
    
  /** density a */
  protected double m_density_a = Double.NaN;
  
  /** density b */
  protected double m_density_b = Double.NaN;
  
  /** total density maximum */
  protected double m_densityMax = Double.NaN;
   
  /** split path eg. "LLLRLR" */
  String m_splitPath = null;
  
  /** split history */
  StringBuffer m_splitHistory = new StringBuffer("");

  /** reference to outest bin */
  protected MultiBin m_outestBin = null;
  
  /** reference to parent bin */
  protected MultiBin m_parentBin = null;
  
  /** number of test instances that fall into bin (also used for a (neg) instances)*/
  protected double m_numInst = 0.0;

  /** number of test b (pos) instances that fall into bin*/
  protected double m_numB_Inst = 0.0;

   /** weight of bin = number of train instances that fall into that bin*/
  protected double m_weight = 0.0;
 
  /** test value for weight, counting all being valid */
  public double m_testWeight = 0.0;
   
  /** median values */
  protected double [] m_medianValue = null;

  /** total number of instances of the histogram */
  protected double m_totalNum = -1.0;

  /** total number of negative instances of the histogram */
  protected double m_totalA_Num = -1.0;

  /** total number of positive instances of the histogram */
  protected double m_totalB_Num = -1.0;

  /**B**/
  /** total number of instances for illegal cut computation */
  protected double m_numInstForIllCut = -1.0;

  /** total length of the histogram */
  protected double [] m_totalLen;
 
  /** number of instances in the bin */
  //protected double [] m_numInst;
  
  /** class index */
  protected int m_classIndex = -1;

  /** total maximum values*/
  protected double [] m_MAXValue;

  /** total range */
  protected double [] m_RANGE;

  /** total minimum values*/
  protected double [] m_MINValue;
  
  /** maximum value of the bin, maxValue is standard to be printed to the right*/
  protected double [] m_maxValue;

  /** flag if max is including leftEnd points */
  protected boolean [] m_maxIncl;

  /** minimum value of the bin, minValue is standard to be printed to the left */
  protected double [] m_minValue;

  /** volume of the bin */
  protected double m_volume;
 
  /** total volume - universe */
  protected double m_totalVolume;
  
  /** flag if min is including leftEnd points */
  protected boolean [] m_minIncl;

  /** flag if the bin contains an illegal cut */
  protected boolean m_illegalCut;

  /** loglk of one test instance in this bin */
  protected double m_oneLoglk = Double.NaN;

  /** train LLK of this bin (weight instead of numinst) */
  protected double m_weightLLK = Double.NaN;

  /** uniform noise level in this bin */
  protected double m_alpha;

  /** stores the density */
  //protected double m_density = -1.0;

  /** stores the likelihood */
  // protected double m_likelihood = -1.0;

  // pointer to corresponding tree node
  protected Tree m_tree = null;
     
  /** Constructor */
  public MultiBin() {
  }

  /** Constructor */
  public MultiBin(MultiBin bin, GlobalSplitData spInfo, Split split, 
      double numInstForIllCut, double alpha, boolean left, Instances globalData) {
    
    m_compareMethod = bin.m_compareMethod;
    m_outestBin = bin.m_outestBin;
    m_parentBin = bin.m_parentBin;

    m_data = spInfo.data;
    m_globalData = globalData;
    
    if (left) {
      m_tree = bin.getTree().leftNode;
      bin.getTree().leftNode.bin = this;
    } else {
      m_tree = bin.getTree().rightNode;
      bin.getTree().rightNode.bin = this;
    }
    
    m_splitDepth = split.splitDepth; 
    m_lastCutAttr = split.attrIndex;
     
    m_splitHistory.append(bin.m_splitHistory + " " + m_lastCutAttr
	+ ", " + split.cutValue + ", ");
    if (left) {
      m_splitHistory.append("L."); 
      m_splitPath = new String(bin.m_splitPath + "L");
    } else {
      m_splitHistory.append("R.");
      m_splitPath = new String(bin.m_splitPath + "R");
    }
      
    m_classIndex = spInfo.classIndex;

    m_MINValue = spInfo.MINValue;
    m_MAXValue = spInfo.MAXValue;
    m_medianValue = spInfo.medianValue;
    m_RANGE = spInfo.range;
    
    // info from upper bin
    int len = bin.m_minValue.length;
    m_numAttr = len;
    
    m_usedAttr = new boolean[len];
    System.arraycopy(bin.m_usedAttr, 0, m_usedAttr, 0, len);
    
    m_minValue = new double[len];
    System.arraycopy(bin.m_minValue, 0, m_minValue, 0, len);
    m_maxValue = new double[len];
    System.arraycopy(bin.m_maxValue, 0, m_maxValue, 0, len);
    m_minIncl = new boolean[len];
    System.arraycopy(bin.m_minIncl, 0, m_minIncl, 0, len);
    m_maxIncl = new boolean[len];
    System.arraycopy(bin.m_maxIncl, 0, m_maxIncl, 0, len);
    
    m_totalVolume = bin.m_totalVolume;
 
    // info from global info
    m_totalLen = new double[len];
    System.arraycopy(spInfo.bigL, 0, m_totalLen, 0, len);
    m_totalNum = spInfo.bigN;
    
    // one length was cutoff and take the right num
    // m_numInst = new boolean[len];
    if (left) {
      m_maxValue[m_lastCutAttr] = split.cutValue;
      m_maxIncl[m_lastCutAttr] = !split.rightFlag;
      //m_numInst[m_lastCutAttr] = split.leftNum; // is the weight
      m_weight = split.leftNum;
    } else {
      m_minValue[m_lastCutAttr] = split.cutValue;
      m_minIncl[m_lastCutAttr] = split.rightFlag;
      //m_numInst[m_lastCutAttr] = split.rightNum; // is the weight
      m_weight = split.rightNum;
    }
    
    m_volume = computeVolume();
    m_numInstForIllCut = numInstForIllCut;
    m_alpha = alpha;
    
    // valid list
    boolean [] valid = bin.getValid();
    m_valid = new boolean[valid.length];
    //DBO.pln("MultiBin-split-left "+split.leftNum+" split-right "+split.rightNum);
    System.arraycopy(valid, 0, m_valid, 0, m_valid.length);
    m_valid = MultiBinningUtils.splitFromValid(m_valid, split.cutValue, 
        split.rightFlag, split.attrIndex, left, spInfo.data, m_globalData);
    
    m_testWeight = testWeight(m_valid);
    
    
    
    
    
    
    
    //DBO.pln("MultiBin-aftersplit\n" +this.fullToString());
    if (m_classIndex > -1) {
      
    }
  }
  
  private int testWeight(boolean [] valid) {
    int weight = 0;
    for (int i = 0; i < valid.length; i++) {
      if (valid[i]) weight++;
    }
    return weight;
  }
 
  /**
   * Constructor that sets a root node
   * @param spInfo all the global info for split
   */
  public MultiBin(GlobalSplitData spInfo, Instances globalData, int compareMethod) {
    
    m_compareMethod = compareMethod;
    m_data = spInfo.data;
    m_globalData = globalData;
    m_parentBin = null;     
    m_outestBin = this;
    
    // referenz to corresponding tree node
    m_tree = spInfo.tree;
    spInfo.tree.bin = this;
    
    // make a root bin
    m_splitDepth = 0; 
    //m_begin = 0;
    m_splitPath = new String("");
    m_lastCutAttr = -1;
    
    m_classIndex = spInfo.classIndex;

    // info from upper bin 
    int len = spInfo.MINValue.length;
    m_numAttr = len;

    m_usedAttr = new boolean[len];
    System.arraycopy(spInfo.usedAttr, 0, m_usedAttr, 0, len);
  
    m_MINValue = spInfo.MINValue;
    m_MAXValue = spInfo.MAXValue;
    m_medianValue = spInfo.medianValue;
    m_RANGE = spInfo.range;
    //m_numInst = len;
    //m_MINValue = new double[len];
    //System.arraycopy(spInfo.MINValue, 0, m_MINValue, 0, len);
    //m_MAXValue = new double[len];
    //System.arraycopy(spInfo.MAXValue, 0, m_MAXValue, 0, len);
    m_minValue = new double[len];
    System.arraycopy(spInfo.MINValue, 0, m_minValue, 0, len);
    m_maxValue = new double[len];
    System.arraycopy(spInfo.MAXValue, 0, m_maxValue, 0, len);
    m_totalLen = new double[len];
    System.arraycopy(spInfo.bigL, 0, m_totalLen, 0, len);
    m_minIncl = new boolean[len];
    m_maxIncl = new boolean[len];
    //m_totalLen = new double[len];
    for (int i = 0; i < m_minIncl.length; i++) {
      m_minIncl[i] = true;
      m_maxIncl[i] = true;
    }
    m_volume = computeVolume();
    m_totalVolume = m_volume;
    m_totalNum = spInfo.bigN;
    //for (int i = 1; i < len; i++) {
    //  m_numInst[i] = m_totalNum;
    //}
    m_weight = m_totalNum;
    m_numInstForIllCut = spInfo.numInstForIllCut;
    m_alpha = spInfo.alpha;

    // valid list, all are valid
    m_valid = new boolean[(int) m_weight];
    for (int i = 0; i < m_totalNum; i++) {
      m_valid[i] = true;
    }
  }

  /** Constructor 
   * 
   * 
   */
  public MultiBin(MultiBin bin,
      //GlobalSplitData spInfo, 
      	Split split, 
      //double numInstForIllCut, double alpha, 
      	boolean left) {
    //Instances globalData) {
    
    m_compareMethod = bin.m_compareMethod;

    m_data = bin.m_data;
    m_globalData = bin.m_globalData;
         
    if (left) {
      m_tree = bin.getTree().leftNode;
      bin.getTree().leftNode.bin = this;
    } else {
      m_tree = bin.getTree().rightNode;
      bin.getTree().rightNode.bin = this;
    }
    
    m_splitDepth = split.splitDepth; 
    m_lastCutAttr = split.attrIndex;
     
    m_splitHistory.append(bin.m_splitHistory + " " + m_lastCutAttr
	+ ", " + split.cutValue + ", ");
    if (left) {
      m_splitHistory.append("L."); 
      m_splitPath = new String(bin.m_splitPath + "L");
    } else {
      m_splitHistory.append("R.");
      m_splitPath = new String(bin.m_splitPath + "R");
    }
     
    //todom_classIndex = spInfo.classIndex;

    m_MINValue = bin.m_MINValue;
    m_MAXValue = bin.m_MAXValue;
    m_medianValue = bin.m_medianValue;
    
    // info from upper bin
    int len = bin.m_minValue.length;
    m_numAttr = len;
    
    m_usedAttr = new boolean[len];
    System.arraycopy(bin.m_usedAttr, 0, m_usedAttr, 0, len);
    
    m_minValue = new double[len];
    System.arraycopy(bin.m_minValue, 0, m_minValue, 0, len);
    m_maxValue = new double[len];
    System.arraycopy(bin.m_maxValue, 0, m_maxValue, 0, len);
    m_RANGE = new double[len];
    System.arraycopy(bin.m_RANGE, 0, m_RANGE, 0, len);  
    m_minIncl = new boolean[len];
    System.arraycopy(bin.m_minIncl, 0, m_minIncl, 0, len);
    m_maxIncl = new boolean[len];
    System.arraycopy(bin.m_maxIncl, 0, m_maxIncl, 0, len);
    
    m_totalVolume = bin.m_totalVolume;
      
    // info from global info
    m_totalLen = new double[len];
    System.arraycopy(bin.m_totalLen, 0, m_totalLen, 0, len);
    m_totalNum = bin.m_totalNum;
    
    // one length was cutoff and take the right num
    // m_numInst = new boolean[len];
    if (left) {
      m_volume = computeVolume();
      //DBO.pln("m_maxValue[m_lastCutAttr] "+m_maxValue[m_lastCutAttr] + " volume " +m_volume);
      m_maxValue[m_lastCutAttr] = split.cutValue;
      m_maxIncl[m_lastCutAttr] = !split.rightFlag;
      //m_numInst[m_lastCutAttr] = split.leftNum; // is the weight
      m_weight = split.leftNum;
    } else {
      m_volume = computeVolume();
      //DBO.pln("m_minValue[m_lastCutAttr] "+m_minValue[m_lastCutAttr] + " volume " +m_volume);
      m_minValue[m_lastCutAttr] = split.cutValue;
      m_minIncl[m_lastCutAttr] = split.rightFlag;
      //m_numInst[m_lastCutAttr] = split.rightNum; // is the weight
      m_weight = split.rightNum;
    }
    //m_RANGE[m_lastCutAttr] = m_maxValue[m_lastCutAttr] - m_minValue[m_lastCutAttr];

    m_volume = computeVolume();
    //DBO.pln("new volume " +m_volume);
       
    // take defaults
    m_numInstForIllCut = 0.0;
    m_alpha = 1.0;
    
    // valid list
    boolean [] valid = bin.getValid();
    m_valid = new boolean[valid.length];
    //DBO.pln("MultiBin-split-left "+split.leftNum+" split-right "+split.rightNum);
    System.arraycopy(valid, 0, m_valid, 0, m_valid.length);
    m_valid = MultiBinningUtils.splitFromValid(m_valid, split.cutValue, 
        split.rightFlag, split.attrIndex, left, m_data, m_globalData);
    //DBO.pln("MultiBin-aftersplit\n" +this.fullToString());
    if (m_classIndex > -1) {
      
    }
  }
 
  /**
   * Coputes the volume using the normalized widths
   * @return volume (never larger than 1.0)
   */
  protected double computeVolume() {

    //DBO.pln("computeValume: ");
    double volume = 1.0; 
    int len = m_maxValue.length;
    for (int i = 0; i < len; i++) {
      if (m_usedAttr[i]) {
	if (m_maxValue[i] > m_minValue[i]) {
	  double width = (m_maxValue[i] - m_minValue[i]) / m_RANGE[i];
	   //**DBO.pln("attr "+i+ " width "+width);
	  volume *= width;
	}
      }
    }
    //DBO.pln("\nvolume "+volume);
    return volume;
  }
  
  /* returns the normalized volume 
   * @return the volume (normalized)
   */
  public double getVolume() {
    return computeVolume(); 
  }
  
  public boolean fitsInto(Instance inst) {
    // DBO.pln("fitsinto");
    boolean fits = true;
    int i = 0;
    double val = 0.0;
    while (fits && i < m_numAttr) {
      if (m_usedAttr[i]) {
	val = inst.value(i);
	//double min = m_minValue[i];
	//double max = m_maxValue[i];
	//boolean f1 = val < m_minValue[i];
	//boolean f2 = val > m_maxValue[i];
	double min = m_minValue[i];
	double max = m_maxValue[i];
	 	// if  (m_maxIncl[i]) {
 	if (m_minIncl[i]) {
	  // 6. 12 fits = (val > m_minValue[i]);
	  
	  // 27.11 
 	  fits = (val >= m_minValue[i]);
	} else {
	  fits = (val > m_minValue[i]);          
	}
	if (m_maxIncl[i]) {
 	//if (m_minIncl[i]) {
         // 27.11 
	  fits = (fits && val <= m_maxValue[i]);
	 // 6.12 fits = (fits && val < m_maxValue[i]);
	} else {
	  boolean f = val < m_maxValue[i];
	  fits = (fits && val < m_maxValue[i]);         
	}
	//DBO.p("value "+val+" min "+min+" max "+max);
	//DBO.pln(" fits? "+fits);

      }
      //if (!fits) {
	//DBO.pln("Didn't fit at attribute "+i+" val "+val);
      //}
      i++;
    }
   return fits;
  }
  
  public boolean fitsInside(Instance inst) {
    boolean fits = true;
    int i = 0;
    while (fits && i < m_numAttr) {
      if (m_usedAttr[i]) {
	double val = inst.value(i);
	fits = (val > m_minValue[i]);          
	fits = (fits && val < m_maxValue[i]);         
      }
      i++;
    }
    return fits;
  }
  
  public void changeIntoDiff(double dens_a, double dens_b) {
    m_density_a = dens_a;
    m_density_b = dens_b;
    m_isDifferenceBin = true;
  }

  public void setDiffMax(double max) {
    m_densityMax = max;
  }
  
  public void setMedianValue() {
   m_medianValue[m_lastCutAttr] = getMedianValue();
  }
  
  public double getMedianValue() {
    int median = (int)(m_weight / 2.0);
    double medianValue = Double.NaN;
    for (int i = 0; i < m_minValue.length; i++) {
	int index = 0;
	int j = 0;
	for (; j < m_totalNum && index < median; j++) {
	  if (m_valid[i]) index++;
	}
	medianValue = (m_data[m_lastCutAttr]).instance(j).value(0);
      }
    
    return medianValue;
  }
  
  public Instance getRepresentative() {
    double [] attVals = new double[m_minValue.length];
    for (int i = 0; i < m_minValue.length; i++) {
      attVals[i] = (m_maxValue[i] + m_minValue[i]) / 2.0;    
    }
    Instance inst = new Instance(1.0, attVals);
    return inst;
  }
  
  /**
   * Adds a test instance into the bin.
   */
  public void addInst() {
    m_numInst = m_numInst + 1.0;
  }
  
  /**
   * Add instance with specific weight
   * @param weight weight of the instance being added
   */
  public void addInst(double weight) {
    m_numInst = m_numInst + weight;
  }
  
  /**
   * Add instance with specific weight to b instances
   * @param weight weight of the instance being added
   */
  public void addB_Inst(double weight) {
    m_numB_Inst = m_numB_Inst + weight;
  }
  
  /**
   * Add TRAIN instance with specific weight
   * @param weight weight of the instance being added
   */
  public void addInstAsTrain(double weight) {
    m_weight = m_weight + weight;
  }
  
  /**
   * Gets the number of test instances in the bin.
   * @return numInst
   */
  public double getNumInst() {
    return m_numInst;
  }
 
  /**
   * Gets the number of test instances in the bin.
   * @return numInst
   */
  public double getNumB_Inst() {
    return m_numB_Inst;
  }
  

  /**
   * Gets the percentage to A instances compared to total test (A + B) instances
   * @return percentage of A instances to test instances
   */
  public double getAofAllPercent() {
    double all = m_numInst + m_numB_Inst;
    double p = MultiBinningUtils.percent(all, m_numInst);
    return p;
  }
  
  /**
   * Gets the percentage to B instances compared to total test (A + B) instances
   * @return percentage of B instances to test instances
   */
  public double getBofAllPercent() {
    double numInst = m_numInst;
    double numB_Inst = m_numB_Inst;
    
    double all = m_numInst + m_numB_Inst;
    double p = MultiBinningUtils.percent(all, m_numB_Inst);
    p = p - 50;
    if (p < 0.0) p = 0.0;
    return p;
  }
  
 /**
   * Gets the percentage to B instances compared to total train instances
   * @return percentage of B instances to training instances
   */
  public double getB_Percent() {
    double p = MultiBinningUtils.percent(m_weight, m_numB_Inst);
    return p;
  }
  
  /**
   * Gets the percentage to B instances compared to total train instances
   * @param numInstInBin number of instances in the bag
   * @return percentage of B instances to training instances
   */
  public double getB_Percent(double numInstInBag) {
    double p = MultiBinningUtils.percent(numInstInBag, m_numB_Inst);
    return p;
  }
  
 /**
   * Gets the difference num A instances minus num B instances  
   * @return num_A - num_B
   */
  public double getDiffAB() {
    double p = m_numInst - m_numB_Inst;
    return p;
  }
  
  /**
   * Gets the number of instances in the bin.
   * @return numInst number of instances
   *
  public double getNumInst(int attrIndex) {
    return m_numInst[attrIndex];
  }

  /**
   * Sets the number of instances in that bin.
   * @param numInst new number of instances
   *
  public void setNumInst(int attrIndex, double numInst) {
    m_numInst[attrIndex] = numInst;
  }*/

  /**
   * Gets the valid list for the bin.
   * @return the valid list for the bin
   */
  public boolean [] getValid() {
    return m_valid;
  }

  /**
   * Gets the attribute index of the previous cut.
   * @return the attributes index
   */
  public int getLastCutAttr() {
    return m_lastCutAttr;
  }

  /**
   * Gets the depth of the bin.
   * @return depth of the bin
   */
  public int getSplitDepth() {
    return m_splitDepth;
  }

  /**
   * Sets the depth of that bin.
   * @param index new depth of the bin
   */
  public void setSplitPath(String path) {
    m_splitPath = path;
  }
  
  /**
   * Gets the split path of the bin.
   * @return split path of the bin
   */
  public String getSplitPath() {
    return m_splitPath;
  }

  /**
   * Sets the spli path of that bin.
   * @param String new split path of the bin
   */
  public void setSplitDepth(int depth) {
    m_splitDepth = depth;
  }

  /**
   * Gets the total length of range.
   * @return total length of range
   */
  public double getTotalLen(int attrIndex) {
    return m_totalLen[attrIndex];
  }
  

  /**
   * Sets the number of test instances in that bin to zero.
   */
  public void emptyBin() {
    //for (int i = 0; i < m_baglessModel.numAttributes(); i++) {
    //  m_numInst[i] = 0.0;
    //}
    m_numInst = 0;
    m_oneLoglk = Double.NaN;
  }

  /**
   * Sets the number of b instances in that bin to zero.
   */
  public void emptyB_Bin() {
    //for (int i = 0; i < m_baglessModel.numAttributes(); i++) {
    //  m_numInst[i] = 0.0;
    //}
    m_numB_Inst = 0;
    //m_oneLoglk = Double.NaN;
  }

  /**
   * Sets the number of TRAIN instances in that bin to zero.
   */
  public void emptyBinAsTrain() {
    m_weight = 0.0;
    m_oneLoglk = Double.NaN;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String weightTipText() {
    return "Specify the weight of that bin.";
  }

  /**
   * Gets the width the bin.
   * @attrIndex index of the attribute
   * @return maximum - minimum
   */
  public double getWidth(int attrIndex) {
    return m_maxValue[attrIndex] - m_minValue[attrIndex];
  }

  /**
   * Gets the weight the bin.
   * @return weight
   */
  public double getWeight() {
    return m_weight;
  }

  /**
   * Sets the weight that bin.
   * @param weight new weight
   */
  public void setWeight(double weight) {
    m_weight = weight;
  }

  /**
   * Sets the total length of the histogram the bin is part of.
   * If set the percentage length can be computed
   * @param len the total length of the histogram
   */
  public void setTotalLen(int attrIndex, double len) {
    m_totalLen[attrIndex] = len;
  }

  /**
   * Sets the total number of instances of the whole histogram the bin is part of.
   * If set the percentage of instances can be computed
   * @param num the total number of instances of the histogram
   */
  public void setTotalNum(double num) {
    m_totalNum = num;
  }

  /**
   * Gets the total number of instances of the whole histogram the bin is part of.
   * @return the total number of instances of the histogram
   */
  public double getTotalNum() {
    return m_totalNum;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String maxValueTipText() {
    return "Specify range upper value.";
  }

  /**
   * Gets the right border of that bin.
   * @param attrIndex index of the attribute used
   * @return maxValue right border of the bin
   */
  public double getMaxValue(int attrIndex) {
    return m_maxValue[attrIndex];
  }

  /**
   * Sets the right border of that bin.
   * @param attrIndex index of the attribute used
   * @param maxValue right border of the bin
   */
  public void setMaxValue(int attrIndex, double maxValue) {
    m_maxValue[attrIndex] = maxValue;
  }

  /**
   * Gets the rightt borders of that bin.
   * @return all right (maximal) borders of the bin
   */
  public double [] getMaxValues( ) {
    return m_maxValue;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String minValueTipText() {
    return "Specify range lower value.";
  }

  /**
   * Gets the left border of that bin.
   * @return minValue left border of the bin
   */
  public double getMinValue(int attrIndex) {
    return m_minValue[attrIndex];
  }

  /**
   * Sets the left border of that bin.
   * @param minValue left border of the bin
   */
  public void setMinValue(int attrIndex, double minValue) {
    m_minValue[attrIndex] = minValue;
  }

  /**
   * Gets the left border of that bin.
   * @return all left (minimal) borders of the bin
   */
  public double [] getMinValues( ) {
    return m_minValue;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui.
   */
  public String minInclTipText() {
    return "Specify if border at minimum includes border point.";
  }

  /**
   * Gets the setting if minimum border is open.
   * @return minopen flag if minimum border is open
   */
  public boolean getMinIncl(int attrIndex) {
    return m_minIncl[attrIndex];
  }

  /**
   * Sets if the  minimum border is open.
   * @param minOpen
   */
  public void setMinIncl(int attrIndex, boolean minIncl) {
    m_minIncl[attrIndex] = minIncl;
  }

  /**
   * Gets the setting of all  minimum borders.
   * @return all flags if minimum border is open
   */
  public boolean [] getMinIncls() {
    return m_minIncl;
  }

  /**
   * Gets the flag is the bin was an illegal cut
   * @return minopen flag if minimum border is open
   */
  public boolean getIllegalCut() {
    return m_illegalCut;
  }

  /**
   * Sets if the bin has an illegal cut.
   * @param flag the new value of the flag
   */
  public void setIllegalCut(boolean flag) {
    m_illegalCut = flag;
  }

  /**
   * Tests itself if the bin has an illegal cut and then sets the flag..
   * @param flag the new value of the flag
   */
  public void setIllegalCut() {
    double totalNum = m_totalNum;
    if (m_numInstForIllCut > 0.0) {
      totalNum = m_numInstForIllCut;
    }
    // check all attributes if there is an illegal cut
    for (int i = 0; i < m_minValue.length; i++) {
      m_illegalCut = (BinningUtils.isIllegalCut((int) m_weight, 
            m_maxValue[i] - m_minValue[i], m_totalLen[i], totalNum));
      if (m_illegalCut) break;
    }
   }
   
  public void setTotalLen(double [] newLens) {
    if (newLens == null) {
      m_totalLen = null;
      return;
    }
    m_totalLen = new double[newLens.length];
    System.arraycopy(newLens, 0, m_totalLen, 0, newLens.length);
  }

  /**
   * Set all number of instances.
   * @param newV new values
   *
  public void setNumInst(int [] newV) {
    if (newV == null) {
      m_numInst = null;
      return;
    }
    m_numInst = new double[newV.length];
    System.arraycopy(newV, 0, m_numInst, 0, newV.length);
  }*/

  /**
   * Set all min values.
   * @param newV new values
   */
 public void setMinValue(double [] newV) {
    if (newV == null) {
      m_minValue = null;
      return;
    }
    m_minValue = new double[newV.length];
    System.arraycopy(newV, 0, m_minValue, 0, newV.length);
  }

  /**
   * Set all max values.
   * @param newV new values
   */
  public void setMaxValue(double [] newV) {
    if (newV == null) {
      m_maxValue = null;
      return;
    }
    m_maxValue = new double[newV.length];
    System.arraycopy(newV, 0, m_maxValue, 0, newV.length);
  }
   
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui.
   */
  public String maxInclTipText() {
    return "Specify if border at minimum includes border point.";
  }

  /**
   * Gets the setting if maximum border is including border point.
   * @return maxOpen flag if maximum border is including
   */
  public boolean getMaxIncl(int attrIndex) {
    return m_maxIncl[attrIndex];
  }

  /**
   * Sets if the maximum border is including border point.
   * @param maxIncl flag if the maximum border is including
   */
  public void setMaxIncl(int attrIndex, boolean maxIncl) {
    m_maxIncl[attrIndex] = maxIncl;
  }

  /**
   * Gets the setting of all maximum borders.
   * @return all flags if maximum border is open
   */
  public boolean [] getMaxIncls() {
    return m_maxIncl;
  }
  
  /**
   * Returns reference to corresponding tree
   */
  public Tree getTree() {
    return m_tree;
  }
  
  /**
   * Returns the leftBegin offset in the bin
   */
 // public int getBegin() {
 //   return m_begin;
 // }
  
  /**
   * Get probability.
   * @return the probability
   * @exception Exception if likelihood is not computeable
   */
  public double getProbability() throws Exception{

//  Oops.pln("totalNum "+m_totalNum );
//  Oops.pln("weight "+m_weight );
    double prob  = m_weight / m_totalNum;
    return prob;
  }

  /**
   * Get density of the multidimensional density function.
   * @return the density
    */
  public double getDensity() {
    
    double dens = 0.0;
    //double weight = m_weight + m_alpha * m_volume / m_totalVolume;
    //dens = weight / (m_volume * (m_totalNum + m_alpha));
    //DBO.pln("dens "+dens);
    //DBO.pln("weight "+m_weight);
    dens = m_weight / (m_volume / m_totalVolume);
    //DBO.pln("dens2 "+dens);
    return dens;
  }

  /**
   * Get density of the multidimensional density function.
   * @return the density
    */
  public double getDivDensity() {
    
    double dens = getB_Density() - getA_Density();
    return dens;
  }
  
  public void setA_BtotalNums(double num_A, double num_B) {
    m_totalA_Num = num_A;
    m_totalB_Num = num_B;
    m_totalNum = num_A + num_B;
  }

  /**
   * Get density of the multidimensional density function.
   * @return the density
    */
  public double getA_Density() {
    
    double dens = 0.0;
    //dens = weight / (m_volume * (m_totalNum + m_alpha));
    //DBO.pln("dens "+dens);
    
    dens = m_numInst / (m_volume * m_totalA_Num);
    //dens = m_numInst / (m_volume * m_totalNum);
       
    //DBO.pln("dens2 "+dens);
    return dens;
  }

  /**
   * Get density of the multidimensional density function.
   * @return the density
    */
  public double getB_Density() {
    
    double dens = 0.0;
    dens = m_numB_Inst / (m_volume * m_totalB_Num);
    //dens = m_numB_Inst / (m_volume * m_totalNum);
    return dens;
  }

  /**
   * Get density of the multidimensional density function.
   * @param real just to differ from the one that doesn't take total density
   * @return the density
    */
  public double getA_Density(boolean real) {
    
    double dens = 0.0;
    dens = m_numInst / (m_volume * m_totalA_Num);
    return dens;
  }

  /**
   * Get density of the multidimensional density function.
   * @param real just to differ from the one that doesn't take total density
   * @return the density
   */
  public double getB_Density(boolean real) {
    
    double dens = 0.0;
    dens = m_numB_Inst / (m_volume * m_totalB_Num);
    return dens;
  }
  
  /**
   * Get loglikelihood with one testinstance in the bin.
   * @return the loglikelihood
    */
  public double getOneLoglk( ) {

//  Oops.pln("totalLen "+m_totalLen );
//  Oops.pln("numInst "+m_numInst);
//  Oops.pln("weight "+m_weight );
//  Oops.pln("alpha "+m_alpha );
 
    if (Double.isNaN(m_oneLoglk)) {
      double weight = m_weight + m_alpha * m_volume / m_totalVolume;
      double x = (weight) / (m_volume * (m_totalNum + m_alpha));
      m_oneLoglk = Math.log((weight) / (m_volume * (m_totalNum + m_alpha)));
    }
    return m_oneLoglk;
  }

  /**
   * Get loglikelihood with the weight as numinst
   * @return the loglikelihood
   * @exception Exception if likelihood is not computeable
   */
  public double getAttrLoglk(int attrIndex) {

    if (m_weight == 0)
      return 0.0;
    double width = m_maxValue[attrIndex] - m_minValue[attrIndex];
    double density = m_weight / (width * m_totalNum);
    double llk  = (m_weight) * (Math.log(density));
    return llk;
  }

  /**
   * For debug purposes: all information of a bin
   * @return info string
   */
  public String debToString() {
    StringBuffer text = new StringBuffer("#| ");

   text.append("splitDepth "+ m_splitDepth+"\n");

  /** split path */
   text.append("splitPath "+ m_splitPath+"\n");

  /** number of instances in the bin */
  // text.append("numInst "+ m_numInst+"\n");

  /** weight of bin */
   text.append("weight "+ m_weight+"\n");

  /** total length of the histogram */
   text.append("totalLen "+ m_totalLen+"\n");

  /** total number of instances of the histogram */
   text.append("totalNum "+ m_totalNum+"\n");

  /** maximum value of the bin, maxValue is standard to be printed to the right*/
   text.append("maxValue "+ m_maxValue+"\n");

  /** flag if max is including leftEnd points */
   text.append("maxIncl "+ m_maxIncl+"\n");

  /** minimum value of the bin, minValue is standard to be printed to the left */
   text.append("minValue "+ m_minValue+"\n");
 
  /** flag if min is including leftEnd points */
   text.append("minIncl "+ m_minIncl+"\n");
 
  /** flag if the bin contains an illegal cut */
   text.append("illegalCut "+ m_illegalCut+"\n");

   /** oneloglk of this bin */
   text.append("oneLoglk "+ m_oneLoglk+"\n");

  /** uniform noise level in this bin */
   text.append("alpha "+ m_alpha+"\n");

  /** stores the density */
   //text.append("density "+ m_density+"\n");

  return text.toString();
  }

  /**
   * Print information about number of instances in a bin
   * @return string with info about the bin
   */
  public String toNumString() {
    StringBuffer text = new StringBuffer(" a= "+m_numInst+" b= "+ m_numB_Inst+"\n");
    return text.toString();    
  }
  
  /**
   * Print main information of a bin
   * @return string with info about the bin
   */
  public String toString() {
    
    int attr = m_lastCutAttr;
    StringBuffer text = new StringBuffer(dimensionsToString() + "#|"+m_lastCutAttr+"| ");
    
    if (attr >= 0) {   
      if (m_minIncl[attr]) { 
        text.append("["); } else { text.append("("); 
        }
      if (m_minValue[attr] == m_MINValue[attr]) text.append("*");    
      text.append(""+Utils.doubleToString(m_minValue[attr], 6)+
          ", "+Utils.doubleToString(m_maxValue[attr], 6));
      if (m_maxValue[attr] == m_MAXValue[attr]) text.append("*");
      if (m_maxIncl[attr]) { 
        text.append("]"); } else { text.append(")"); 
        }
      
      text.append(" | ");
      if (m_totalLen[attr] > 0.0) {
        text.append(" "+percentString(m_totalLen[attr], m_maxValue[attr] - m_minValue[attr], 6)+"%");
      } else {
        text.append("       ");
      }
    }
    text.append(" ||");
    text.append(" Weight "+Utils.doubleToString(m_weight, 6, 6));
    text.append(" | ");
    if (m_totalNum > 0.0) {
      text.append(" "+percentString(m_totalNum, m_weight, 6)+"%");
    }
    text.append(" ||");
    text.append(" Volume "+Utils.doubleToString(m_volume, 6, 6));
     
    try {
      text.append(" | Num " + Utils.doubleToString(m_numInst, 6, 6));
      text.append(" | Dens " + Utils.doubleToString(getDensity() ,6));
      text.append(" | Prob " + Utils.doubleToString(getProbability(), 6));
      //text.append(" | " + Utils.doubleToString((m_weight/9.0) - m_numInst, 6) );
      //text.append(" |L " + Utils.doubleToString(getLoglikelihood(int attrIndex) ,6));
      //text.append(" |LO " + Utils.doubleToString(getLoglikeliForOne(int attrIndex) ,6));
    } catch (Exception ex) {}
    text.append(" |\n");
    return text.toString();
  }

  /**
   * Print main information of a bin
   * @return string with info about the bin
   */
  public String toNiceString() {
    
    StringBuffer text = new StringBuffer("CutPath "+m_splitPath);
     text.append(" ||");
    text.append(" Weight "+Utils.doubleToString(m_weight, 6, 6));
    text.append(" | ");
    if (m_totalNum > 0.0) {
      text.append(" "+percentString(m_totalNum, m_weight, 6)+"%");
    }
    text.append(" ||");
    text.append(" Volume "+Utils.doubleToString(m_volume, 6, 6));
     
    try {
      text.append(" | Num A " + Utils.doubleToString(m_numInst, 6, 6));
      text.append(" | Num B " + Utils.doubleToString(m_numB_Inst, 6, 6));
      text.append(" | Dens " + Utils.doubleToString(getDensity() ,6));
    } catch (Exception ex) {}
    text.append(" |\n");
    return text.toString();
  }

 public String dimensionsToString() {
    StringBuffer text = new StringBuffer("#|"+m_lastCutAttr+"| ");
    
    for (int attr = 0; attr < m_MINValue.length; attr++) {
      if (m_minIncl[attr]) { text.append("["); } else { text.append("("); }
      if (m_minValue[attr] == m_MINValue[attr]) text.append("*"); 
      
      text.append(""+Utils.doubleToString(m_minValue[attr], 6)+
          ", "+Utils.doubleToString(m_maxValue[attr], 6));
      
      if (m_maxValue[attr] == m_MAXValue[attr]) text.append("*");
      if (m_maxIncl[attr]) { text.append("]"); } else { text.append(")"); }
    }
    text.append(" |\n");
    return text.toString();
  }

  
  public String getAttrPictBlock(int attrIndex) {
    StringBuffer text = new StringBuffer("");
    int min = 0;
    double dmin = m_minValue[attrIndex];
    double dmax = m_maxValue[attrIndex];
    double dMIN = m_MINValue[attrIndex];
    double dMAX = m_MAXValue[attrIndex];
    double range = dMAX-dMIN;
       
    if (m_minValue[attrIndex] > m_MINValue[attrIndex]) {
      min = (int) Math.rint(0.5 + ((((dmin - dMIN) * 100.0) / range) / 10.0));
    } else {
      min = 0;
    }
    
    int max = 10;
    if (m_maxValue[attrIndex] < m_MAXValue[attrIndex]) {
      max = (int) Math.rint(0.5 + ((((dmax - dMIN) * 100.0) / range) / 10.0));
    } else {
      max = 10;
    
    }
    
    
    text.append("[");
       int i = 0;
    for (; i < min; i++) {
      text.append(".");
    }
    if (min == max) {
      text.append("I");
    }
    for (; i < max; i++) {
      text.append("X");
    }
    for (; i < 10; i++) {
      text.append(".");
    }
    text.append("]");
       
    return text.toString();
  }
    
  /**
   * 
   * @param sum_a total of A instances (negativ)
   * @param sum_b total of B instances (positive)
   * @param maxDens maximal density (a or b)
   * @return
   */
  public String getABString(double maxDens) {
    StringBuffer text = new StringBuffer("");
    //double dens_a = getA_Density(true);
    //double dens_b = getB_Density(true);
    double dens_a = getA_Density();
    double dens_b = getB_Density();
    int d_a = (int) Math.rint(0.5 + ((dens_a * 100.0) / maxDens) / 10.0);
    int d_b = (int) Math.rint(0.5 + ((dens_b * 100.0) / maxDens) / 10.0);
    double p_a =(dens_a * 100.0) / maxDens;
    double p_b =(dens_b * 100.0) / maxDens;
    
    // zerolevel
    boolean empty = false;
    if (dens_a == 0.0 && dens_b == 0.0) {
      text.append("-");
      empty = true;
    }
    else {
      if (dens_a == 0) text.append("A");
      else if (dens_b == 0) text.append("B");
      else text.append("0");
    }
    
    // middle part
    if (empty) text.append("..........");
    else {
      StringBuffer letter1 = new StringBuffer("");
      StringBuffer letter2 = new StringBuffer("");
      int minMax = d_a;
      int maxMax = d_b;
      if (dens_b < dens_a) {
	letter1.append("b");
	letter2.append("a");
	minMax = d_b;
	maxMax = d_a;
      } else {
	letter1.append("a");
	letter2.append("b");   
      }
      for (int i = 0; i < minMax; i++) { text.append(letter1); }
      for (int i = minMax; i < maxMax; i++) { text.append(letter2); }
      for (int i = maxMax; i < 10; i++) { text.append("."); }
    }

    // leftEnd part
    if (dens_a > dens_b) {
      text.append("A");
    } else {
      if (dens_a < dens_b) {
	      text.append("B");
      } else {
	text.append("-");
      }
    }
    //text.append(" A:" + p_a + " B:" + p_b);
    return text.toString();
  }
  
  public String getABString() {
    StringBuffer text = new StringBuffer("");
     int d_a = (int) Math.rint(0.5 + ((m_density_a * 100.0) / m_densityMax) / 10.0);
    int d_b = (int) Math.rint(0.5 + ((m_density_b * 100.0) / m_densityMax) / 10.0);
    double p_a =(m_density_a * 100.0) / m_densityMax;
    double p_b =(m_density_b * 100.0) / m_densityMax;
    
    // zerolevel
    boolean empty = false;
    if (m_density_a == 0.0 && m_density_b == 0.0) {
      text.append("-");
      empty = true;
    }
    else {
      if (m_density_a == 0) text.append("A");
      else if (m_density_b == 0) text.append("B");
      else text.append("0");
    }
    
    // middle part
    if (empty) text.append("..........");
    else {
      StringBuffer letter1 = new StringBuffer("");
      StringBuffer letter2 = new StringBuffer("");
      int minMax = d_a;
      int maxMax = d_b;
      if (m_density_b < m_density_a) {
	letter1.append("b");
	letter2.append("a");
	minMax = d_b;
	maxMax = d_a;
      } else {
	letter1.append("a");
	letter2.append("b");   
      }
      for (int i = 0; i < minMax; i++) { text.append(letter1); }
      for (int i = minMax; i < maxMax; i++) { text.append(letter2); }
      for (int i = maxMax; i < 10; i++) { text.append("."); }
    }
    
  
    // leftEnd part
    if (m_density_a > m_density_b) {
      text.append("A");
    } else {
      if (m_density_a < m_density_b) {
	text.append("B");
      } else {
	text.append("-");
      }
    }
    //text.append(" A:" + p_a + " B:" + p_b);
    return text.toString();
  }

  public String toABString(boolean ab, boolean allAttr) {
    StringBuffer text = new StringBuffer("");
    if (ab) text.append(getABString());
    if (allAttr) text.append(getAllAttrString(18,1));
    text.append("\n");
    return text.toString();
  }
  
  public String toPictStringRow(boolean ab, boolean shDensity, 
      boolean shWeight, boolean shVolume, boolean oneline, 
      double maxDensity, double maxABDensity) {
    double density = 0.0;
    StringBuffer text = new StringBuffer("");
    
    // make the ab - string
    if (ab) text.append(getABString(maxABDensity));
    if (shDensity) {
      density = getDensity();
      text.append(" Dns:"+percentPictString(maxDensity, density, true));
	//+" "+percentString(m_totalNum, m_weight, 6) +" "+ m_weight+" "+m_totalNum);
    }
    ////text.append(" totalnum " + m_totalNum + " m_weight "+m_weight );
    if (shWeight) text.append(" Ins:"+percentPictString(m_totalNum, m_weight, true));
    double weight = m_weight;
    double totalNum = m_totalNum;
	////+" "+percentString(m_totalNum, m_weight, 6) +" "+ m_weight+" "+m_totalNum
	////);
    if (shVolume) text.append(" Vol:"+percentPictString(m_totalVolume, m_volume, true));
	////+ " " 	 +percentString(m_totalVolume, m_volume, 6)
	////+ " "+m_volume+" "+m_totalVolume
	////);
    //text.append(" Dens:"+percent(maxDensity, density));
    double h = MultiBinningUtils.percent(m_totalVolume, m_volume);
    double l = Math.log(h) / Math.log(10.0);
    //text.append(" Vol:"+ h +" "+ l +   "\n");
    if (!oneline) text.append("\n");
    return text.toString();
  }
  
  public String getAllAttrString(int totalLength, int afterComma) {
    StringBuffer text = new StringBuffer("");
    for (int i = 0; i < m_numAttr; i++) {
      text.append(""+ getAttrString(i, totalLength, afterComma));
    }
   
    return text.toString();
  }
  
  public String getAllAttrOnlyCutString(int totalLength, int afterComma,
      Instances dataModel) {
    StringBuffer text = new StringBuffer("");
    for (int i = 0; i < m_numAttr; i++) {
      String attrStr = getAttrOnlyCutString(i, totalLength, afterComma);
      if (attrStr != null) {
	text.append("" + i + ": " + attrStr
	    + dataModel.attribute(i).name() + "\n");
      }
    }  
    return text.toString();
  }

  public String getAllAttrPictBlock() {
    StringBuffer text = new StringBuffer("");
    int indent = 0;
    int j = 0;
    for (int i = 0; i < m_numAttr; i++) {
      text.append(""+ getAttrPictBlock(i)+"");
      j++;
      if (j == 5) {
	j = 0;
	indent += 2;
	text.append("\n");
	for (int ii = 0; ii < indent; ii++) {
	  text.append(" ");
	}
      }
    }
    return text.toString();
  }

  public String getAttrString(int attr, int totalLength, int afterComma) {
    StringBuffer text = new StringBuffer(" ");

    if (m_minIncl[attr]) { 
      text.append("["); } else { text.append("("); 
      }
    if (m_minValue[attr] == m_MINValue[attr]) text.append("*");    
    text.append("" + Utils.doubleToString(m_minValue[attr], afterComma) +
	", " + Utils.doubleToString(m_maxValue[attr], afterComma));
    if (m_maxValue[attr] == m_MAXValue[attr]) text.append("*");
    if (m_maxIncl[attr]) { 
      text.append("]"); } else { text.append(")"); 
      }
    if (text.length() > totalLength) {
      text = new StringBuffer("");
      for (int i = 0; i < totalLength; i++) {
	text.append("-");
      }
    } else {
      for (int i = text.length(); i < totalLength; i++) {
	text.append(" ");
      }     
    }
    return text.toString();
  }

  /**
   * Returns 
   * @param attrIndex
   */
  public String getAttrOnlyCutString(int attr, int totalLength, int afterComma) {
    StringBuffer text = new StringBuffer(" ");

    // no string if no cut in attributes range
    if ((m_minValue[attr] == m_MINValue[attr]) && 
	(m_maxValue[attr] == m_MAXValue[attr])) {
      return null;
    }
    
    if (m_minIncl[attr]) { 
      text.append("["); } else { text.append("("); 
      }
    if (m_minValue[attr] == m_MINValue[attr]) text.append("*");    
    text.append("" + Utils.doubleToString(m_minValue[attr], afterComma) +
	", " + Utils.doubleToString(m_maxValue[attr], afterComma));
    if (m_maxValue[attr] == m_MAXValue[attr]) text.append("*");
    if (m_maxIncl[attr]) { 
      text.append("]"); } else { text.append(")"); 
      }
    if (text.length() > totalLength) {
      text = new StringBuffer("");
      for (int i = 0; i < totalLength; i++) {
	String out = text.toString();
	out = out.substring(0, totalLength - 1);
	return out;
      }
    } else {
      for (int i = text.length(); i < totalLength; i++) {
	text.append(" ");
      }     
    }
    return text.toString();
  }

  /**
   * Print main information of a difference bin
   * @return string with info about the bin
   */
  public String toDiffString() {
    
    int attr = m_lastCutAttr;
    StringBuffer text = new StringBuffer("#|"+m_lastCutAttr+"| ");
    
    if (attr >= 0) {   
      if (m_minIncl[attr]) { 
        text.append("["); } else { text.append("("); 
        }
      if (m_minValue[attr] == m_MINValue[attr]) text.append("*");    
      text.append("" + Utils.doubleToString(m_minValue[attr], 6) +
          ", " + Utils.doubleToString(m_maxValue[attr], 6));
      if (m_maxValue[attr] == m_MAXValue[attr]) text.append("*");
      if (m_maxIncl[attr]) { 
        text.append("]"); } else { text.append(")"); 
        }
      
      text.append(" | ");
      if (m_totalLen[attr] > 0.0) {
        text.append(" "+percentString(m_totalLen[attr], m_maxValue[attr] - m_minValue[attr], 6)+"%");
      } else {
        text.append("       ");
      }
    }
    text.append(" ||");
    double diff = Math.abs(m_density_a - m_density_b);
    text.append(" Diff "+Utils.doubleToString(diff, 6));
    text.append(" ||");
    text.append(" Diff% "+percentString(m_densityMax, diff, 6));
    text.append(" ||");
   
    if (m_density_a < m_density_b) {
      text.append(" max = b");
      text.append(" ||");
      text.append(" Min "+Utils.doubleToString(m_density_a, 6));
      text.append(" ||");
   } else {
      text.append(" max = a");
      text.append(" ||");
      text.append(" Min "+Utils.doubleToString(m_density_b, 6));
      text.append(" ||");
    }
   //text.append(" Volume "+Utils.doubleToString(m_volume, 6, 6));
     
    text.append(" || "+getABString()+  "\n");
    return text.toString();
  }
 
  /**
   * Print information of all dimensions of the bin
   * @return string with info about the bin
   */
  public String fullBinToString() {
    int aIndex = 0;
    int iIndex = 1;
    
    int attr = m_lastCutAttr;
    StringBuffer text = new StringBuffer(m_splitHistory);
    text.append(" || ");
    text.append(" Weight: "+Utils.doubleToString(m_weight, 6, 0));
    text.append(" | ");
    if (m_totalNum > 0.0) {
      text.append(" "+percentString(m_totalNum, m_weight, 6)+"%");
    }
    text.append(" || \n");


    for (attr = 0; attr < m_numAttr; attr++) {
      if (m_usedAttr[attr]) {
	if (m_minIncl[attr]) { 
	  text.append("["); }
	else { text.append("("); }    
	if (m_minValue[attr] == m_MINValue[attr]) text.append("*");
	text.append(""+Utils.doubleToString(m_minValue[attr], 6)+
	    ", "+Utils.doubleToString(m_maxValue[attr], 6));
	if (m_maxValue[attr] == m_MAXValue[attr]) text.append("*");
	if (m_maxIncl[attr]) { 
	  text.append("]"); }
	else { text.append(")"); }
      }
    }
    text.append("\n\n");

    return text.toString();
  }

  /**
   * Print info for a rectangle plot
   * * @param xDim attribute index of the x dimension
   * * @param yDim attribute index of the y dimension
   * @return string with rectangle plot
   */
  public String rectangleString(int xDim, int yDim) {
 
    StringBuffer text = new StringBuffer("");

    // check if any of the 2 ranges cut
    boolean isCut = true;
    if (m_minValue[xDim] == m_MINValue[xDim]) {
      if (m_maxValue[xDim] == m_MAXValue[xDim]) {
	if (m_minValue[yDim] == m_MINValue[yDim]) {
	  if (m_maxValue[yDim] == m_MAXValue[yDim]) {
	    isCut = false;
	  }
	}
      }
    }
    // return empty string if none of the 2 ranges is cut
    if (!isCut) return text.toString();
    // else ...
    
    // write rectangle string
    
    text.append("" + m_minValue[xDim] + "," + m_minValue[yDim]);
    text.append("" + m_maxValue[xDim] + "," + m_minValue[yDim]);
    text.append("" + m_maxValue[xDim] + "," + m_maxValue[yDim]);
    text.append("" + m_minValue[xDim] + "," + m_maxValue[yDim]);
    text.append("" + m_minValue[xDim] + "," + m_minValue[yDim]);
 
    return text.toString();
  }

  /**
   * Print information of all dimensions of the bin
   * @return string with info about the bin
   */
  public String fullBinToPictBlock(boolean showDiff) {
    StringBuffer text = new StringBuffer("");
    //m_splitHistory);
    //text.append(" || ");
    //text.append(" weight: "+Utils.doubleToString(m_weight, 6, 0));
    //if (m_totalNum > 0.0) {
    //  text.append(" "+percentString(m_totalNum, m_weight, 6)+"%");
    //}
    //if (showDiff) text.append(" " + getABString());
    text.append("\n\n");
 
    text.append(getAllAttrPictBlock());
    text.append("\n\n");

    return text.toString();
  }

  /**
   * Print information of all dimensions of the bin
   * @param dataModel the data model that supports the partition
   * @return string with info about the bin
   */
  public String fullBinToRulesText(Instances dataModel) {
    StringBuffer text = new StringBuffer("\n");
    for (int attr = 0; attr < m_numAttr; attr++) {
      if (!(m_minValue[attr] == m_MINValue[attr])) {
	text.append(" " + dataModel.attribute(attr).name() + " >");
	if (m_minIncl[attr]) { text.append("= "); }
	text.append("" + m_minValue[attr]);
	text.append("\n");  
	if (!(m_maxValue[attr] == m_MAXValue[attr])) {
	  text.append(" and");
	}
     }
      if (!(m_maxValue[attr] == m_MAXValue[attr])) {
	text.append(" " + dataModel.attribute(attr).name() + " <");
	if (m_maxIncl[attr]) { text.append("= "); }
	text.append("" + m_maxValue[attr]);
	text.append("\n");  
      }
    }
    text.append("\n");
    return text.toString();
  }

  /**
   * Print information of all dimensions that have been cut
   * @param onlyCut if true take only the cut attributes
   * @return string with ranges of the bin
   */
  public String rangesToString(boolean onlyCut) {
      
    StringBuffer text = new StringBuffer("");
      
    for (int attr = 0; attr < m_numAttr; attr++) {
      // if attribute is a used attribute
      if (m_usedAttr[attr]) {
	
	// if attribute is cut
	if (!onlyCut ||
	    !(m_minValue[attr] == m_MINValue[attr] && m_maxValue[attr] == m_MAXValue[attr])) {
	  if (m_minIncl[attr]) { 
	    text.append("["); }
	  else { text.append("("); }    
	  if (m_minValue[attr] == m_MINValue[attr]) text.append("*");
	  text.append(""+Utils.doubleToString(m_minValue[attr], 6)+
	      ", "+Utils.doubleToString(m_maxValue[attr], 6));
	  if (m_maxValue[attr] == m_MAXValue[attr]) text.append("*");
	  if (m_maxIncl[attr]) { 
	    text.append("]"); } 
	  else { text.append(")"); }
	}
      }
    }
    return text.toString();
  }
  
  /**
   * Print information of all instances in the bin
   * @return string with info about the bin
   */
  public String fullResultsToString() {
    int aIndex = 0;
    int iIndex = 1;
    
    StringBuffer text = new StringBuffer(fullBinToString());
    int attr = m_lastCutAttr;
     
    text.append(">>>>>> list all instances, num should be "+m_weight +"::\n");
    if (attr >= 0) {
      int num = 0;
      for (int i = 0; i < m_valid.length; i++) {
        int index = (int)m_data[attr].instance(i).value(iIndex);
        if (m_valid[index]) {
          text.append(""+index+" |"+m_globalData.instance(index));
          text.append(" |"+index+"\n");
          num++;
        }
      }
      text.append(" | num= "+num);
    }
    text.append("  leftEnd fullResultsTostring\n\n");
    return text.toString();
  }
  
  /**
   * Print verbose information of a bin
   * @return string with info about the bin
   */
  public String verboseToString() {
    
    StringBuffer text = new StringBuffer("#| "+m_splitPath+"\n#| ");
    
    for (int i = 0; i < m_minValue.length; i++) {
      if (m_minValue[i] > m_outestBin.m_minValue[i] || 
          m_maxValue[i] < m_outestBin.m_maxValue[i]) {  
        if (m_minIncl[i]) { 
          text.append("["); } else { text.append("("); 
          }
        text.append(""+Utils.doubleToString(m_minValue[i], 6)+
            ", "+Utils.doubleToString(m_maxValue[i], 6));
        if (m_maxIncl[i]) { 
          text.append("]"); } else { text.append(")"); 
          }
     
        text.append(" | ");
        if (m_totalLen[i] > 0.0) {
          text.append(" "+percentString(m_totalLen[i], m_maxValue[i] - m_minValue[i], 6)+"%");
        } else {
          text.append("       ");
        }
        text.append(" ||");
      }
    }
    text.append("      "+Utils.doubleToString(m_weight, 6, 0));
    text.append(" | ");
    if (m_totalNum > 0.0) {
      text.append(" "+percentString(m_totalNum, m_weight, 6)+"%");
    }
    
    try {
      //text.append(" | " + m_numInst);
      //text.append(" |D " + Utils.doubleToString(getDensity(int attrIndex) ,6));
      text.append(" |P " + Utils.doubleToString(getProbability() ,6));
      //text.append(" | " + Utils.doubleToString((m_weight/9.0) - m_numInst, 6) );
      //text.append(" |L " + Utils.doubleToString(getLoglikelihood(int attrIndex) ,6));
      //text.append(" |LO " + Utils.doubleToString(getLoglikeliForOne(int attrIndex) ,6));
    } catch (Exception ex) {}
    text.append(" |\n");
    return text.toString();
  }
  
 /**
  * Returns the percentage as a string
  * @param total the total value
  * @param sub the part of the value that should be transformed to %
  * @param len the maximal length of the string
  * @return the string representing the percentage value
  */
  protected String percentString(double total, double sub, int len) {
    double p = (sub * 100.0) / total;
    String str = Utils.doubleToString(p, 2)+"       ";
    return str.substring(0, len - 1);
  }
  
    
 /**
   * Returns the percentage as a string
   * @param total the total value
   * @param sub the part of the value that should be transformed to %
   * @param len the maximal length of the string
   * @return the string representing the percentage value
   */
   protected String percentPictString(double total, double sub, boolean shSmaller) {
     StringBuffer text = new StringBuffer("");
     int max = 0;
     boolean small = false;
     double p = 0.0;
     
     if (sub > 0.0) {
       if (sub == total) {
	 max = 10;
       } else {
	 p = (sub * 100.0) / total;
	 if (shSmaller && p < 0.1) {
	   small = true;
	 } else {  
	   max = (int) Math.rint(0.5 + (p / 10.0));
	 }
       }
       if (small) {
	 if (p < 0.0000001) {  
	   int e = (int) Math.rint(Math.log(p)/Math.log(10.0) - 1.0);
	   String dots = ".......";
	   String t =  "[<1E"+e;

	   text.append(t + dots.substring(0, 11 - t.length()) + "]");
	   //text.append("[<0.0000001]");
	 }
	 else if (p < 0.000001)   text.append("[<0.000001.]");
	 else if (p < 0.00001)    text.append("[<0.00001..]");
	 else if (p < 0.0001)     text.append("[<0.0001...]");
	 else if (p < 0.001)      text.append("[<0.001....]");
	 else if (p < 0.01)       text.append("[<0.01.....]");
	 else if (p < 0.1)        text.append("[<0.1......]");
	 /*else if (p < 10.0) {
	   String dots = ".......";
	   String t =  "["+Utils.doubleToString(p,2);
	   text.append(t + dots.substring(0, 11 - t.length()) + "]");
	 }*/
       } else {
	 text.append("[");
	 int i = 0;
	 for (; i < max; i++) {
	   text.append("X");
	 }
	 for (; i < 10; i++) {
	   text.append(".");
	 }
	 text.append("]");
       }

     
     } else {
       // value is zero
       text.append("[          ]");
     }
     return text.toString();
   }
     
    
    
  /**
   * Tests whether the current bin object is equal to another
   * bin object
   * @param obj the object to compare against
   * @return true if the two objects are equal
   */
  public boolean equals(Object obj) {
    
    if ((obj == null) || !(obj.getClass().equals(this.getClass()))) {
      return false;
    }
    MultiBin cmp = (MultiBin) obj;
    if (m_splitDepth != cmp.m_splitDepth) return false;
    if (m_splitPath  != cmp.m_splitPath) return false;
    if (m_weight  != cmp.m_weight) return false;
    if (m_totalLen  != cmp.m_totalLen) return false;
    if (m_totalNum  != cmp.m_totalNum) return false;
    if (m_numInstForIllCut  != cmp.m_numInstForIllCut) return false;
    for (int i = 0; i < m_maxValue.length; i++) {
      if (m_usedAttr[i] != cmp.m_usedAttr[i]) return false;
      if (m_maxValue[i]  != cmp.m_maxValue[i]) return false;
      if (m_maxIncl[i]  != cmp.m_maxIncl[i]) return false;
      if (m_minValue[i]  != cmp.m_minValue[i]) return false;
      if (m_minIncl[i]  != cmp.m_minIncl[i]) return false;
      //if (m_numInst[i]  != cmp.m_numInst[i]) return false;
    }
    if (m_illegalCut  != cmp.m_illegalCut) return false;
     if (m_weightLLK  != cmp.m_weightLLK) return false;
    if ( m_alpha != cmp.m_alpha) return false;
    if (m_oneLoglk != cmp.m_oneLoglk) return false;
    
    return true;
  }
  
  /**
   * Compare method for interface Comparator
   * this is first bin
   * @param b second bin
   * @return -1, 0 or 1 if a < b, == b or > b
   */
  public int compareTo(MultiBin b) {
    double aDens = 0.0;
    double bDens = 0.0;
    MultiBin a = this;
    switch (m_compareMethod) {
    case 1:
      aDens = ((MultiBin)a).getDensity();
      bDens = ((MultiBin)b).getDensity();
      break;
    case 2:
      aDens = ((MultiBin)a).getB_Density() - ((MultiBin)a).getA_Density();
      bDens = ((MultiBin)b).getB_Density() - ((MultiBin)b).getA_Density();
      break;
    }
    if (bDens < aDens) return -1; 
    else {
      if (aDens == bDens) return 0;
    }
    return 1;
  }
  
  /**
   * Computes a centre point 
   * @param debug
   * @param data
   * @return
   */
  public Instance getCentrePoint(boolean debug, Instances data) {
    Instance inst = null;
    int numInst = data.numInstances();
    int numAtt = data.numAttributes();
    double [] attrSum = new double[numAtt];
    int realNumAtt = 0;
    for (int i = 0; i < numAtt; i++) {
      attrSum[i] = 0.0;
      if (m_usedAttr[i])
	realNumAtt++;
    }
    ////!!!! works only with normalized and standardized 
    ////!!!! otherwise sums too large!!!
    int numFitting = 0;
    for (int i = 0; i < numInst; i++) {
      inst = data.instance(i);
      if (fitsInto(inst)) {
	numFitting++;
	if (debug) 
	  DBO.pln("I: " + inst);
	for (int j = 0; j < numAtt; j++) {
	  if (m_usedAttr[j])
	    attrSum[j] += inst.value(j); 
	}
      }
    }
    for (int j = 0; j < numAtt; j++) {
      if (m_usedAttr[j])
	attrSum[j] = attrSum[j] / numFitting;
    }
    
    // make new instance
    Instance newInst = new Instance(1.0, attrSum);
    //tests
    if (debug) {
      if (fitsInto(newInst))
        DBO.pln("centrepoint fits into  "+newInst);
      else {
	DBO.pln("centrepoint DOESNT FIT "+newInst);
	DBO.pln(allMins());
	DBO.pln(allMaxs());
      }	
    }
    return newInst;
  }
 
  /**
   * Selects randomly one of the train instances in the bin as centre point
   * @param debug
   * @param data
   * @param random
   * @return
   */
  public Instance getCentrePoint(boolean debug, Instances data, Random random) {
    Instance inst = null;
    int numInst = data.numInstances();
    int numAtt = data.numAttributes();
    double [] attrSum = new double[numAtt];
    Vector<Instance> fittingInstances = new Vector();

    ////!!!! works only with normalized and standardized 
    ////!!!! otherwise sums too large!!!
    for (int i = 0; i < numInst; i++) {
      inst = data.instance(i);
      if (fitsInto(inst)) {
	fittingInstances.add(inst);
	if (debug) 
	  DBO.pln("I: " + inst);
      }
    }
 
    if (fittingInstances.size() == 0) return null;
    int choice = random.nextInt(fittingInstances.size());
    Instance retInst = fittingInstances.elementAt(choice);
    return retInst;
  }

  public String allMaxs () {
    StringBuffer txt = new StringBuffer("maxs: ");
    int numAtt = m_maxValue.length;
    for (int i = 0; i < numAtt; i++) {
      txt = txt.append(m_maxValue[i] + ",");
    }
    return txt.toString();
  }

  public String allMins () {
    StringBuffer txt = new StringBuffer("mins: ");
    int numAtt = m_minValue.length;
    for (int i = 0; i < numAtt; i++) {
      txt = txt.append(m_minValue[i] + ",");
    }
    return txt.toString();
  }

  public Instance getCentrePointReal(boolean debug, Instances data) {
    Instance inst = null;
    int numInst = data.numInstances();
     for (int i = 0; i < numInst; i++) {
      inst = data.instance(i);
      if (fitsInto(inst)) 
	return inst;
    }
     return null;
  }
  
  
  /**
   * Main method for testing this class.
   *
   * @param argv should contain arguments to the filter: use -h for help
   */
  public static void main(String [] argv) {
    
    try {
      MultiBin bin = new MultiBin();
      DBO.pln("" + bin.toString()); 
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
  }
}
