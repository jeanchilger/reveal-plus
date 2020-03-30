//26.5
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
 *    Bin.java
 *    Copyright (C) 2004 Gabi Schmidberger
 *
 *    Class represents a bin of a discretized numerical Variable
 */

package weka.filters.unsupervised.attribute;

import java.io.Serializable;
import weka.core.Utils;
import weka.estimators.BinningUtils;
import weka.core.Debug.DBO;

/** 
 *
 * Class represents a bin. Bin is produced by a discretizing algorithm<p>
 *
 * @author Gabi Schmidberger (gabi@cs.waikato.ac.nz)
 * @version $Revision: 0.0 $
 */

public class Bin implements Serializable{

  /** split depth from which the bin originates from */
  private int m_splitDepth;

  /** split path */
  String m_splitPath = null;

  /** number of instances in the bin */
  private double m_numInst;

  /** weight of bin */
  private double m_weight = 0.0;

  /** leftBegin index in sorted dataset */
  private int m_begin;

  /** leftEnd index in sorted dataset */
  private int m_end;

  /** leftBegin index in list of possible cuts */
  private int m_beginCuts;

  /** leftEnd index in list of possible cuts */
  private int m_endCuts;

  /** total length of the histogram */
  private double m_totalLen = -1.0;

  /** total number of instances of the histogram */
  private double m_totalNum = -1.0;

  /** total number of instances for illegal cut computation */
  private double m_numInstForIllCut = -1.0;

  /** maximum value of the bin, maxValue is standard to be printed to the right*/
  private double m_maxValue;

  /** flag if max is including leftEnd points */
  private boolean m_maxIncl;

  /** minimum value of the bin, minValue is standard to be printed to the left */
  private double m_minValue;

  /** flag if min is including leftEnd points */
  private boolean m_minIncl;

  /** flag if the bin contains an illegal cut */
  private boolean m_illegalCut;

  /** entropy of this bin */
  private double m_entropy;

  /** train LLK of this bin (weight instead of numinst) */
  private double m_weightLLK = Double.NaN;

  /** uniform noise level in this bin */
  private double m_alpha;

  /** stores the density */
  //private double m_density = -1.0;

  /** stores the likelihood */
  private double m_likelihood = -1.0;

  /** Constructor */
  public Bin() {
  }

  /** Constructor that sets all values */
  public Bin(int splitDepth,
	     double totalNum, double numInstForIllCut, double totalLen,
	     double num, int begin, int end,
	     double min, boolean minIn, double max, boolean maxIn,
	     double entropy, double alpha) {
    m_splitDepth = splitDepth;
    m_totalNum = totalNum;
    m_numInstForIllCut = numInstForIllCut;
    m_totalLen = totalLen;
    m_numInst = num;
    m_weight = num;
    m_begin = begin;
    m_end = end;
    m_minValue = min;
    m_minIncl = minIn;
    m_maxValue = max;
    m_maxIncl = maxIn;
    m_entropy = entropy;
    m_alpha = alpha;
    m_likelihood = getLikelihood();
    // m_density = getDensity();
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numInstTipText() {
    return "Specify the number of instances in that bin.";
  }

//   /**
//    * Gets the index of the bin.
//    * @return index of the bin
//    */
//   public int getIndex() {
//     return m_index;
//   }

//   /**
//    * Sets the index of that bin.
//    * @param index new index of the bin
//    */
//   public void setIndex(int index) {
//     m_index = index;
//   }

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
  public double getTotalLen() {
    return m_totalLen;
  }

  /**
   * Gets the number of instances in the bin.
   * @return numInst number of instances
   */
  public double getNumInst() {
    return m_numInst;
  }

  /**
   * Sets the number of instances in that bin.
   * @param numInst new number of instances
   */
  public void setNumInst(double numInst) {
    m_numInst = numInst;
  }

  /**
   * Sets the number of instances in that bin to zero.
   * @param numInst new number of instances
   */
  public void emptyBin() {
    m_numInst = 0.0;
    //m_density = -1.0;
    m_likelihood = -1.0;
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
   * @return maximum - minimum
   */
  public double getWidth() {
    return m_maxValue - m_minValue;
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
  public void setTotalLen(double len) {
    m_totalLen = len;
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
   * @return maxValue right border of the bin
   */
  public double getMaxValue() {
    return m_maxValue;
  }

  /**
   * Sets the right border of that bin.
   * @param maxValue right border of the bin
   */
  public void setMaxValue(double maxValue) {
    m_maxValue = maxValue;
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
  public double getMinValue() {
    return m_minValue;
  }

  /**
   * Sets the left border of that bin.
   * @param minValue left border of the bin
   */
  public void setMinValue(double minValue) {
    m_minValue = minValue;
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
  public boolean getMinIncl() {
    return m_minIncl;
  }

  /**
   * Sets if the  minimum border is open.
   * @param minOpen
   */
  public void setMinIncl(boolean minIncl) {
    m_minIncl = minIncl;
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
     m_illegalCut = (BinningUtils.isIllegalCut((int) m_weight, m_maxValue - m_minValue, 
						 m_totalLen, totalNum));
     double width =  m_maxValue - m_minValue;
     //     DBO.pln("Bin.setIllegalCut "+m_illegalCut+" : "+m_weight+
     //	      " : "+width+" : "+m_totalLen);
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
  public boolean getMaxIncl() {
    return m_maxIncl;
  }

  /**
   * Sets if the maximum border is including border point.
   * @param maxIncl flag if the maximum border is including
   */
  public void setMaxIncl(boolean maxIncl) {
    m_maxIncl = maxIncl;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String beginTipText() {
    return "Specify the first index if the dataset is sorted.";
  }

  /**
   * Gets the leftBegin index of the instances (if the dataset is sorted).
   * @return leftBegin index of the instances
   */
  public int getBegin() {
    return m_begin;
  }

  /**
   * Sets the leftBegin index of the instances (if the dataset is sorted).
   * @param leftBegin index of the instances
   */
  public void setBegin(int begin)  {
    m_begin = begin;
  }

  /**
   * Gets the leftBegin index of the instances (if the dataset is sorted).
   * @return leftBegin index of the instances
   */
  public int getBeginCuts() {
    return m_beginCuts;
  }

  /**
   * Sets the leftBegin index of the instances (if the dataset is sorted).
   * @param leftBegin index of the instances
   */
  public void setBeginCuts(int begin)  {
    m_beginCuts = begin;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String endTipText() {
    return "Specify the last index, if the dataset is sorted.";
  }

  /**
   * Gets the leftEnd index of the instances (if the dataset is sorted).
   * @return leftEnd index of the instances
   */
  public int getEnd() {
    return m_end;
  }

  /**
   * Sets the leftEnd index of the instances (if the dataset is sorted).
   * @param leftEnd index of the instances
   */
  public void setEnd(int end)  {
    m_end = end;
  }

  /**
   * Gets the leftEnd index in the cut list
   * @return leftEnd index of the instances
   */
  public int getEndCuts() {
    return m_endCuts;
  }

  /**
   * Sets the leftEnd index of the instances (if the dataset is sorted).
   * @param leftEnd index of the instances
   */
  public void setEndCuts(int end)  {
    m_endCuts = end;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String entropyTipText() {
    return "Specify the entropy value of the bin.";
  }

  /**
   * Gets the entropy value of the bin.
   * @return entropy value of the bin
   */
  public double getEntropy() {
    return m_entropy;
  }

  /**
   * Sets the entropy value of the bin.
   * @param entropy
   */
  public void setEntropy(double entropy) {
    m_entropy = entropy;
  }

 /**
   * Add one instance to the bin
   */
  public void addInstance () {
    m_numInst++;
  }

  /**
   * Add one instance as weight to the border bin
   * @param left if true it is the left border else the right
   */
  public void addBorderInstance(boolean left, double value) {
    m_numInst++;
    if (left) {
      if (value < m_minValue) {
	m_minValue = value;
      } 
    } else {
      if (value > m_maxValue) {
	m_maxValue = value;
      }
    }
  }

  /**
   * Add one instance as weight to the bin
   */
  public void addWeight () {
    m_weight += 1.0;
  }

  /**
   * Add one instance as weight to the bin
   * @param w the weight to be added
   */
  public void addWeight (double w) {
    m_weight += w;
  }

  /**
   * Substract one instance as weight to the bin
   */
  public void delWeight () {
    m_weight -= 1.0;
  }

  /**
   * Substract one instance as weight to the bin
   * @param w the weight to be substraced
   */
  public void delWeight (double w) {
    m_weight -= w;
  }

  /**
   * Add one instance as weight to the border bin
   * @param left if true it is the left border else the right
   */
  public void addBorderWeight (boolean left, double value) {
    m_weight++;
    if (left) {
      if (value < m_minValue) {
	m_minValue = value;
      } 
    } else {
      if (value > m_maxValue) {
	m_maxValue = value;
      }
    }
  }

  /**
   * Get probability.
   * @return the probability
   * @exception Exception if likelihood is not computeable
   */
  public double getProbability() throws Exception{

//  DBO.pln("totalNum "+m_totalNum );
//  DBO.pln("weight "+m_weight );
    double prob  = m_weight / m_totalNum;
    return prob;
  }

//   /**
//    * Get loglikelihood.
//    * @return the loglikelihood
//    * @exception Exception if likelihood is not computeable
//    */
//   public double getLogProbability() throws Exception{

// //  DBO.pln("totalLen "+m_totalLen );
// //  DBO.pln("numInst "+m_numInst);
// //  DBO.pln("weight "+m_weight );
// //  DBO.pln("alpha "+m_alpha );
//     double width = m_maxValue - m_minValue;

//     double weight = m_weight + m_alpha * width / m_totalLen;
//     double loglk = - (m_numInst) 
//       * Math.log((weight) / (m_totalNum + m_alpha));
//     return loglk;
//   }

  /**
   * Get loglikelihood.
   * @return the loglikelihood
   * @exception Exception if likelihood is not computeable
   */
  public double getLoglikelihood() throws Exception{

//  DBO.pln("totalLen "+m_totalLen );
//  DBO.pln("numInst "+m_numInst);
//  DBO.pln("weight "+m_weight );
//  DBO.pln("alpha "+m_alpha );
    double width = m_maxValue - m_minValue;
    if (width == 0.0) return Double.MAX_VALUE;

    double weight = m_weight + m_alpha * width / m_totalLen;
    double loglk = 0.0;
    loglk = (m_numInst) 
      * Math.log((weight) / (width * (m_totalNum + m_alpha)));
    return loglk;
  }

  /**
   * Get loglikelihood.
   * @return the loglikelihood
   * @exception Exception if likelihood is not computeable
   */
  public double getLoglikeliForOne() throws Exception{

//  DBO.pln("totalLen "+m_totalLen );
//  DBO.pln("numInst "+m_numInst);
//  DBO.pln("weight "+m_weight );
//  DBO.pln("alpha "+m_alpha );
    double width = m_maxValue - m_minValue;
    if (width == 0.0) return Double.MAX_VALUE;

    double weight = m_weight + m_alpha * width / m_totalLen;
    double loglk =  (1.0) 
      * Math.log((weight) / (width * (m_totalNum + m_alpha)));
    return loglk;
  }


  /**
   * Get loglikelihood.
   * @return the loglikelihood
   * @exception Exception if likelihood is not computeable
   */
  public double getLOOCVLoglikelihood() throws Exception{

    //DBO.pln("getLOOCVLoglikelihood");

    double width = m_maxValue - m_minValue;
    if (width == 0.0) return Double.MAX_VALUE;
    
    if (m_weight == 0.0) return 0.0;
    
    double weight = m_weight - 1.0 + (m_alpha * width / m_totalLen);
    double loglk =  Math.log((weight) / (width * (m_totalNum + m_alpha)));
//      DBO.pln("totalLen "+m_totalLen );
//      DBO.pln("totalLen "+m_totalNum );
//      DBO.pln("numInst "+m_numInst);
//      DBO.pln("weight "+m_weight );
//      DBO.pln("widtht "+width );
//      DBO.pln("alpha "+m_alpha );
//      DBO.pln("loglk "+loglk );
    return loglk;
  }


  /**
   * Get loglikelihood with the weight as numinst
   * @return the loglikelihood
   * @exception Exception if likelihood is not computeable
   */
  public double getWeightLoglikelihood() {

    if (m_weight == 0)
      return 0.0;
    double width = m_maxValue - m_minValue;
    //DBO.pln("getWeightLoglikelihood num "+m_weight+" width "+width+" totalNum "+m_totalNum);
    //       if (width == 0.0) return Double.MAX_VALUE;
    //       double weight = m_weight + m_alpha * width / m_totalLen;
    //       double loglk = 0.0;
    //       loglk =  (m_weight) 
    // 	* Math.log((m_weight) / (width * (m_totalNum)));
    double density = m_weight / (width * m_totalNum);
    double llk  = (m_weight) * (Math.log(density));
    //DBO.pln("WeightLoglikelihood "+llk);
    return llk;
  }

  /**
   * Get likelihood.
   * @return the likelihood 
   */
  public double getLikelihood() {
 
    if (m_weight == 0.0) return 0.0;
      
    double width = m_maxValue - m_minValue;
    if (width == 0.0) return Double.MAX_VALUE;
    
    m_likelihood =  m_numInst * m_weight / (width * m_totalNum);
    return m_likelihood;
  }

  /**
   * Get squared likelihood.
   * @return the squared likelihood 
   */
  public double getSquaredLikelihood() {
 
    double lk = getLikelihood();
    return lk * lk;
  }

  /**
   * Get area underneath the squared likelihood.
   * @return the area underneath the squared likelihood 
   */
  public double getSquaredArea() {
 
    if (m_weight == 0.0) return 0.0;
    
    double width = m_maxValue - m_minValue;
//    DBO.pln("width "+width);
//    DBO.pln("weight "+m_weight );
    double llk =  m_weight / (width * m_totalNum);
    double squared = llk * llk;
    squared = squared * width;
    //    DBO.pln("squared "+squared  );
    
    return squared;
  }

  /**
   * Get density.
   * @return the density 
   */
  public double getDensity() {
    //DBO.pln("Bin-getDensity");

//    DBO.pln("totalLen "+m_totalLen );
//    DBO.pln("totalNum "+m_totalNum );
//    DBO.pln("numInst "+m_numInst);
//    DBO.pln("weight "+m_weight );
//    DBO.pln("alpha "+m_alpha );
    
    double width = m_maxValue - m_minValue;
//    DBO.pln("width "+width);
    if (width == 0.0) return Double.MAX_VALUE;
    
    double weight = m_weight + m_alpha * width / m_totalLen;
    double totalNum = m_totalNum + m_alpha;
    double density =  weight / (width * totalNum);
    //m_density = density;
    
    // DBO.pln("density "+m_density);
    return density;
  }

  /**
   * Get density using num of instances instead of weight.
   * @return the inst density 
   */
  public double getInstDensity() {
    double width = m_maxValue - m_minValue;
    if (width == 0.0) return Double.MAX_VALUE;
    
    double numInst = m_numInst + m_alpha * width / m_totalLen;
    double totalNum = m_totalNum + m_alpha;
    double density =  numInst / (width * totalNum);
    return density;
  }

  /**
   * Get error.
   * @return the error in number of instances 
   */
  public double getError() throws Exception{
    double err = Math.abs(m_weight/9.0 - m_numInst);
    return err;
  }

  /**
   * For debug purposes: all information of a bin
   */
  public String debToString() {
    StringBuffer text = new StringBuffer("#| ");

   text.append("splitDepth "+ m_splitDepth+"\n");

  /** split path */
   text.append("splitPath "+ m_splitPath+"\n");

  /** number of instances in the bin */
   text.append("numInst "+ m_numInst+"\n");

  /** weight of bin */
   text.append("weight "+ m_weight+"\n");

  /** leftBegin index in sorted dataset */
   text.append("leftBegin "+ m_begin+"\n");

  /** leftEnd index in sorted dataset */
   text.append("leftEnd "+ m_end+"\n");

  /** leftBegin index in list of possible cuts */
    text.append("beginCuts "+ m_beginCuts+"\n");

  /** leftEnd index in list of possible cuts */
   text.append("endCuts "+ m_endCuts+"\n");

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

  /** entropy of this bin */
   text.append("entropy "+ m_entropy+"\n");

  /** uniform noise level in this bin */
   text.append("alpha "+ m_alpha+"\n");

  /** stores the density */
   //text.append("density "+ m_density+"\n");

  /** stores the likelihood */
   text.append("likelihood "+ m_likelihood+"\n");

   return text.toString();
  }


  /**
   * Print main information of a bin
   */
  public String toString() {
    StringBuffer text = new StringBuffer("#| ");
    if (m_minIncl) { text.append("["); } else { text.append("("); }
    text.append(""+Utils.doubleToString(m_minValue, 6)+
		", "+Utils.doubleToString(m_maxValue, 6));
    if (m_maxIncl) { text.append("]"); } else { text.append(")"); }

    text.append(" | ");
    if (m_totalLen > 0.0) {
      text.append(" "+percentString(m_totalLen, m_maxValue - m_minValue, 6)+"%");
    } else {
      text.append("       ");
    }
    text.append(" ||");
    text.append("      "+Utils.doubleToString(m_weight, 6, 0));
    text.append(" | ");
   if (m_totalNum > 0.0) {
      text.append(" "+percentString(m_totalNum, m_weight, 6)+"%");
    }
     
    try {
      text.append(" | " + m_numInst);
      text.append(" |D " + Utils.doubleToString(getDensity() ,6));
      text.append(" |P " + Utils.doubleToString(getProbability() ,6));
      //text.append(" | " + Utils.doubleToString((m_weight/9.0) - m_numInst, 6) );
      text.append(" |L " + Utils.doubleToString(getLoglikelihood() ,6));
      text.append(" |LO " + Utils.doubleToString(getLoglikeliForOne() ,6));
    } catch (Exception ex) {}
      text.append(" |\n");
    return text.toString();
  }

  protected String percentString(double total, double sub, int len) {
    double p = (sub * 100.0) / total;
    String str = Utils.doubleToString(p, 2)+"       ";
    return str.substring(0, len - 1);
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
    Bin cmp = (Bin) obj;
    if (m_splitDepth != cmp.m_splitDepth) return false;
    if (m_splitPath  != cmp.m_splitPath) return false;
    if (m_numInst  != cmp.m_numInst) return false;
    if (m_weight  != cmp.m_weight) return false;
    if (m_begin  != cmp. m_begin) return false;
    if (m_end  != cmp.m_end) return false;
    if (m_beginCuts  != cmp.m_beginCuts) return false;
    if (m_endCuts  != cmp.m_endCuts) return false;
    if (m_totalLen  != cmp.m_totalLen) return false;
    if (m_totalNum  != cmp.m_totalNum) return false;
    if (m_numInstForIllCut  != cmp.m_numInstForIllCut) return false;
    if (m_maxValue  != cmp.m_maxValue) return false;
    if (m_maxIncl  != cmp.m_maxIncl) return false;
    if (m_minValue  != cmp.m_minValue) return false;
    if (m_minIncl  != cmp.m_minIncl) return false;
    if (m_illegalCut  != cmp.m_illegalCut) return false;
    if (m_entropy  != cmp.m_entropy) return false;
    if (m_weightLLK  != cmp.m_weightLLK) return false;
    if ( m_alpha != cmp.m_alpha) return false;
    if (m_likelihood  != cmp.m_likelihood) return false;
     
    return true;
  }

  /**
   * Main method for testing this class.
   *
   * @param argv should contain arguments to the filter: use -h for help
   */
  public static void main(String [] argv) {

    try {
      Bin bin = new Bin();
      DBO.pln("" + bin.toString()); 
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
  }

}
