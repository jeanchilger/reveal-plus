package weka.estimators;

import java.io.Serializable;

import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.Debug.DBO;
import weka.estimators.MultiBinningUtils.GlobalSplitData;
import weka.estimators.MultiBinningUtils.Split;
 
public final class AttrTUBEGrid extends AttrTUBE {
    
  /**
	 * 
	 */
	private static final long serialVersionUID = -1742384827471963315L;

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
	 *    AttrTUBEGrid.java
	 *    Copyright (C) 2009 Gabi Schmidberger
	 */
	/** 
	<!-- globalinfo-start -->
	*
	* Abstract class for TUBE-type grid binning estimators.
	*
	<!-- globalinfo-leftEnd -->
	* @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
	* @version $Revision: 1.0 $
	**/

	private class GridInfo implements Serializable {
    
    /**
	 * 
	 */
	private static final long serialVersionUID = -9123006499757210L;
	// number of instances in grid
    int num = -1;
   // index of first value in grid
    int begin = -1;
    // index of last value in grid
    int end = -1;
     // index of first value in next grid
    int nextBegin = -1;
    // first value in Grid
    //double firstValue = Double.NaN;
    // min can be reduced if grid is the first in the range
    double min = Double.NaN;
    // cutValue is max of grid
    double cutValue = Double.NaN;   
    
    boolean allEmpty () { return num == 0; }
    
    boolean lastGrid (double max) { return cutValue >= max; } 
   }
  
 /** number of grid cells */
  private int m_gridNum = -1;
  
  /** leftBegin value of the presplitgrid */
  private double m_gridStart = Double.NaN;
  
  /** distance between the splits with presplit-splitting */
  private double m_gridWidth = Double.NaN;
  
  /**
   * Sets the parameters for grid cutting
   * @exception if the grid cannot be build 
   */
  public void setGrid() throws Exception {
    double num = (double) getGridNum();
    if (num < 1.0) throw new Exception("Grid cannot be initialized for grid cutting");
    m_gridWidth = (m_MAXValue - m_MINValue) / num;
    m_gridStart = m_MINValue + m_gridWidth;
  }
  
  /**
   * prepares an attribute for cutting
   * @param data data set with instances
   * @param attrIndex index of the attribute to be prepared
   */
  public AttrTUBEGrid() {
    
    super();
    
    // file postfix for histogram files and similar
    m_filePostfix = "TUg";  
  }
  
  /**
   * initialize with new data set
   * @param data the dataset to initialize the attribute estimator
   * @param attrIndex the index of the attribute
   * @exception if initialize does not work 
   */
  public void initializeNewData(Instances data, int attrIndex, double min, double max) throws Exception{
    
    super.initializeNewData (data, attrIndex, min, max);
    
    //  prepare grid
    setGrid();
  }
     
  /**
   * Find the cut point in the range with min criteria
   * @param gridBegin left most = leftBegin value of the grid
   * @param bin the bin where the split should be found
   * @param data the data set, used for data model information
   * @param attrIndex the index of the attribute that is discretized
   * @param cutPoints the values order in increasing way
   * @return how much the number of bins has increased in this branch
   */
  protected Split findMinInRange(GlobalSplitData sp, Split split) {  
    
    double minValue = split.bin.getMinValue(m_attrIndex);
    double maxValue = split.bin.getMaxValue(m_attrIndex);         
    if (dbo.dl(D_ABOUTSPLIT)) {
      DBO.pln("find min in range "+minValue+"--"+maxValue);
    }
    if (minValue > maxValue) 
      return null;
    
    double numInst = split.bin.getWeight();
    // don't split empty bin
    if (numInst == 0) 
      return null;
    
    int totalN = (int)sp.bigN;
    double totalL = sp.bigL[m_attrIndex];
    boolean [] binValid = split.bin.getValid();
    
    double leftDist = 0.0;
    double newCriterion = -Double.MAX_VALUE;
    double leftCriterion = 0.0;
    double rightCriterion = 0.0;
    double cutValue = 0.0;   
    double bestCutValue = 0.0;   
    
    // compute first criteria
    double oldCriterion = getCriteriaFromBin(split.bin);
    //dbo.dpln("oldLLK = "+oldLLK);
    
    // prepare split object
    split.attrIndex = m_attrIndex;
    split.newCriterion = -Double.MAX_VALUE;
    split.oldCriterion = oldCriterion;
    split.index = -1;
    
    split.leftDist = 0.0;
    
    int leftNum = 0;
    int rightNum = (int)numInst;

    // find the next grid with the cutValue at the leftEnd of the grid
    GridInfo grid = getFirstGrid(0, (int)numInst, minValue, maxValue, binValid);
    //while (grid.allEmpty() && !grid.lastGrid(maxValue)) {
    //  grid = getNextGrid(grid, maxValue, binValid);
    //}
      
    // no more grid cuts found in range
    if (grid.lastGrid(maxValue)) 
      return split;
    
    leftNum += grid.num;
    rightNum -= grid.num;
    
    cutValue = grid.cutValue;
    leftDist = cutValue - minValue;
    double rightDist = maxValue - cutValue; 
     
    // **************************************************************************
    // check over all instances starting from the first cutpoint within the range
    // from the left range border, cutting at equal distances 
    while (!grid.lastGrid(maxValue)) {
      // get entropy value for this cut
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
          split.lastLeft = grid.end;
          split.firstRight = grid.nextBegin;
          split.cutValue = cutValue;
          bestCutValue = cutValue;
          split.leftDist = leftDist;
          split.rightDist = rightDist;
          split.leftNum = leftNum;
          split.rightNum = rightNum;
          }
      }
      
      // find next cut value
      grid = getNextGrid(grid, maxValue, binValid);
      if (grid.lastGrid(maxValue)) break;
     
      cutValue = grid.cutValue;
      leftNum += grid.num;
      rightNum -= grid.num;
      leftDist = cutValue - minValue;
      rightDist = maxValue - cutValue; 
      // leftEnd of the big while loop
    }
    
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
   * Get the next all empty grid. 
   * Get:
   * leftBegin offset,
   * num of Atoms, number in each atom
   * @param numInst the smallestest value in the range
   * @param minValue the smallestest value in the range
   * @param maxValue the largest value in the range
   * @return the first offset in the range
   */
  protected GridInfo getFirstGrid(int begin, int numInst,
      double minValue, double maxValue, boolean [] binValid) {
    GridInfo grid = new GridInfo();
    
    grid.cutValue = getFirstGridValue(m_gridStart, minValue);
    grid.min = grid.cutValue - m_gridWidth;
    if (grid.min < minValue) grid.min = minValue;
        
    // find first valid value in grid
    int offset = findFirstOffset(grid.min, true, begin, grid.cutValue, binValid);
    grid.begin = offset;
    int offsetForNext = 0;
    if (grid.begin == -1) {
      grid.begin = 0;
      grid.end = -1;
      grid.num = 0;
      return grid;
    } else {
    
      // find last valid value in grid and count numbers
      int [] off_num = findLastOffset(false, grid.begin, grid.cutValue, binValid);
      grid.end = off_num[0];
      grid.num = off_num[1] + 1;
      if (grid.end >= 0)
        offsetForNext = grid.end;
    }
    
    // find first valid in next grid
    offset = findFirstOffset(grid.cutValue, false, offsetForNext, 
        maxValue, binValid);
    grid.nextBegin = offset;
    return grid;
  }
  
  /**
   * Get the next not all empty grid. 
   * Get:
   * leftBegin offset,
   * num of Atoms, number in each atom
   * @param grid the last grid
   * @return the first offset in the range
   */
  protected GridInfo getNextGrid(GridInfo lastGrid, double maxValue, boolean [] binValid) {
    GridInfo grid = new GridInfo();
    
    grid.cutValue = lastGrid.cutValue + m_gridWidth;
    grid.min = lastGrid.cutValue;
          
    if (lastGrid.nextBegin == -1) {
      // no more instances
      grid.begin = 0;
      grid.end = -1;
      grid.nextBegin = -1;
      grid.num = 0;
      return grid;
    }
    
    // find first valid value in grid
    int offset = lastGrid.end;
    if (offset < 0) offset = 0;
    offset = findFirstOffset(grid.min, true, offset, grid.cutValue, binValid);
    grid.begin = offset;
    int offsetForNext = 0;
    if (grid.begin == -1) {
      grid.begin = 0;
      grid.end = -1;
      grid.num = 0;
     } else {    
       // find last valid value in grid and count numbers
       int [] off_num = findLastOffset(false, grid.begin, grid.cutValue, binValid);
       grid.end = off_num[0];
       grid.num = off_num[1] + 1; 
       offsetForNext = grid.end;
     }
           
    // find first valid in next grid
    offset = findFirstOffset(grid.cutValue, false, offsetForNext, 
        maxValue, binValid);
    grid.nextBegin = offset;
   
    return grid;
  }

  /**
   * Get the next cut value, for the gridcut method.
   * @param oldCut the old cut value
   * @return 
   */
  private double getNextCutValue(double oldCut) { 
    
    double cutValue = oldCut + m_gridWidth;
    return cutValue;
  }
  
  /**
   * Get the first leftmost cut value using the grid method.
   * @param gridStart start = left most of grid
   * @param minValue minimal value in range
   * @return the first cut value in the range
   */
  private double getFirstGridValue(double gridStart, double minValue) { 
    double cutValue = gridStart;
    if (cutValue < minValue) {
      double gridNums = Math.ceil((minValue - gridStart) / m_gridWidth); 
      cutValue = gridStart + (gridNums * m_gridWidth);
    }
    if (cutValue <= minValue) cutValue += m_gridWidth;
    return cutValue;
  } 
 
  /**
   * Set number of grid cells for grid cutting
   * @param full int number of gridcells
   */
  public void setGridNum(int num) {
    m_gridNum = num; 
  }
  
  /**
   * Returns number of grid cells for grid cutting 
   * @return number of grid cells for grid cutting
   */
  public int getGridNum() {
    return m_gridNum;
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
    
    // set grid number
    String gridNum = Utils.getOption('G', options);
    if (gridNum.length() > 0) {
      setGridNum(Integer.parseInt(gridNum));
    }
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
