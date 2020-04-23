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
 *    MultiDifference.java
 *    Copyright (C) 2009 Gabi Schmidberger
 */
package weka.estimators;

import java.util.Vector;

import weka.core.Instance;
import weka.core.Instances;
import weka.estimators.MultiBinningUtils.Split;
import weka.estimators.MultiBinningUtils.Tree;

/** 
*
<!-- globalinfo-start -->
* Class builds a multidimensional difference histogram.
<!-- globalinfo-leftEnd -->
*
* @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
* @version $Revision: 1.0 $
*/
public final class MultiDifference 
extends MultiTUBE {
    
  /**
	 * 
	 */
	private static final long serialVersionUID = -6733875138800816505L;

	/** tree built by cutting by both trees */
  protected Tree m_diffTree = null;
  
  /** store the total number of instances */
  int m_totalNum = -1;
   
  /** the bins that result from discretization*/
  //protected Vector m_bins;

  /** list all cutpoints at leftEnd */
  //public static int D_SUMILLCUTS      = 12; // 13
  
  /** list all cutpoints at leftEnd */
  //public static int D_NUMCUTMISE      = 13; // 14
  
 /** trace through precedures */
  //public static int D_TRACE           = 14; // 15
  
  /** look at at each real bin made */
  //public static int D_LOOKATBINS      = 18; // 19
  
  /** probability is zero */
  //public static int D_NULLPROB        = 19; // 20
        
  /** vector to store all new cuts */
  //protected Vector m_splitList = new Vector();
  
    
  //stuff set by options
  
  /** list all cutpoints at leftEnd */
  //protected static final int C_TOOHIGH_EW10      = 1;
  
  /*
   * Constructor
   * Takes first tree and uses second tree to cut again.
   * If transform flag is set bins are transformed into diff-bins
   * @param tree_a the first bin (cut) tree
   * @param tree_b the second bin (cut) tree
   * @param transform flag to transform the bins
   */
  public MultiDifference(int numInst, Tree tree_a, Tree tree_b, boolean transform, MultiTUBE est,
      boolean[] usedAttrs_a, boolean[] usedAttrs_b) {
    
    m_totalNum = numInst;
    int len = usedAttrs_a.length;
    m_usedAttrs = new boolean[len];
    for (int i = 0; i < len; i++) {
      if (usedAttrs_a[i] || usedAttrs_b[i]) {
	m_usedAttrs[i] = true;
      } else
	m_usedAttrs[i] = false;
    }
    
    // starting up
    m_diffTree = tree_a;
    if (tree_a == null) {
      m_diffTree = tree_b;
      // early exit 1 
      return; // result is tree b
    }
    // early exit 2
    if (tree_b == null) return; // result is tree a 
    
    // start with cutting the cut points of b in a
    findDifferenceBins(tree_b);
    
    // gather bins into global variable
    m_bins = MultiBinningUtils.gatherBins(m_diffTree);
    int numBins = m_bins.size();
    m_tree = m_diffTree;
    
    // tidy up bins
    for (int i = 0; i < numBins; i++) {
      MultiBin bin = (MultiBin) m_bins.elementAt(i);
      bin.setTotalNum(m_totalNum);
    }    

    
    if (transform) {
      // change int o difference bins
      changeIntoDiff(m_bins, tree_b);
    }  
    m_originalData = est.m_originalData;
    
  }
  
 /*
  public MultiDifference(Vector bins_a, Instances data_a, Instances data_b) {
    
    // starting up
   Vector bins_b = (Vector)bins_a.clone();
   
   // empty and fill bins a
   MultiBinningUtils.emptyBins(bins_a);
   
   // empty and fill bins b
   MultiBinningUtils.emptyBins(bins_b);

   
   m_bins = bins_a;
    
   // change into difference bins
    changeIntoDiff(m_bins, bins_b);
  }*/

  public MultiDifference(Vector bins, Instances data_a, Instances data_b, MultiTUBE est) {

    // empty bins 
    MultiBinningUtils.emptyBins(bins);

    m_bins = bins;

    // change into difference bins
    changeIntoDiff(m_bins, data_a, data_b);
    
    m_originalData = est.m_originalData;
  }
 
  /**
   * Build difference tree and bins
   * @param tree_b the difference bin
   */  
  private void findDifferenceBins(Tree tree_b) {
    
    if (tree_b == null) return; // result is tree a
             
    // if not leave node take current cut in b and cut through in a
    if (!Double.isNaN(tree_b.cutValue)) {
      //dbo.pln("findDifferenceBins: cut attr " + tree_b.attrIndex + " cut value " + tree_b.cutValue);
      
      cutTree(m_diffTree, tree_b.attrIndex, tree_b.cutValue, tree_b.rightFlag, 0);
    }
    findDifferenceBins(tree_b.leftNode);
    findDifferenceBins(tree_b.rightNode);
    return;
  }
  
  /*
   *  public static class Tree implements Serializable {
   
    int attrIndex = -1;
    double cutValue = Double.NaN;
    boolean rightFlag = true;
    Tree leftNode = null;
    Tree rightNode = null;
    MultiBin bin = null;
  }
  */
  
  private void cutTree(Tree tr, int attrIndex, double cutValue, boolean rightFlag, int splitDepth) {
    boolean noRight = false;
    boolean noLeft = false;
    
    if (tr == null) return;
      
    // is not a leave node
    if (tr.rightNode != null || tr.leftNode != null) {
      
      // check if one of the two sub trees can be left out
      if (attrIndex == tr.attrIndex) {
	if (cutValue <= tr.cutValue) {
	  noRight = true;
	}
	if (cutValue >= tr.cutValue) {
	  noLeft = true;
	}
      }

      if (!noRight && tr.rightNode != null) {
	cutTree(tr.rightNode, attrIndex, cutValue, rightFlag, splitDepth + 1);
      }
      if (!noLeft && tr.leftNode != null) {
	cutTree(tr.leftNode, attrIndex, cutValue, rightFlag, splitDepth + 1);
      }
    } else {
      // is a leave node
      tr.attrIndex = attrIndex;
      tr.cutValue = cutValue;
      tr.rightFlag = rightFlag;
      
      // if leave node, split
      Split split = new Split();
      split.bin = tr.bin;
      split.attrIndex = attrIndex;
      split.rightFlag = rightFlag;
      split.cutValue = cutValue;
      split.splitDepth = splitDepth;

      // widths and numbers
      double width = tr.bin.getMaxValue(attrIndex) - tr.bin.getMinValue(attrIndex);
      split.leftDist = cutValue - tr.bin.getMinValue(attrIndex);
      split.rightDist = tr.bin.getMaxValue(attrIndex) - cutValue;
      double leftPortion = split.leftDist / width;
      double rightPortion = split.rightDist / width;
      //split.leftNum = tr.bin.getNumInst() * leftPortion;
      //split.rightNum = tr.bin.getNumInst() * rightPortion;
      split.leftNum = tr.bin.getWeight() * leftPortion;
      split.rightNum = tr.bin.getWeight() * rightPortion;

      // split.splitNumber = ??;

      // Loglikelihoods
      double criterion = tr.bin.getAttrLoglk(attrIndex);
      split.leftCriterion = criterion * leftPortion;
      split.rightCriterion = criterion * rightPortion;

      split.oldCriterion = criterion;
      split.newCriterion = criterion;

      // split.trainCriterionDiff = irrelevant;

      // index of the instance where it is split
      // split.index = unknown;

      // the middle leftBegin and ends, they differ according to right flag
      // split.lastLeft = unknown;
      // split.firstRight = unknown;

      // the bin in which the split happened
      boolean left = true;
      tr.leftNode = new Tree();
      tr.leftNode.bin = new MultiBin(split.bin, split, left);
      //DBO.pln("left\n"+leftBin.fullResultsToString());

      tr.rightNode = new Tree();
      tr.rightNode.bin = new MultiBin(split.bin, split, !left);
      //DBO.pln("right\n"+rightBin.fullResultsToString()); 
    }
  }
    
/*
 *   public static class Split implements Serializable {
    
    int attrIndex = -1;
    int splitNumber = -1;
    
    // Loglikelihoods
    double leftCriterion = Double.MAX_VALUE;
    double rightCriterion = Double.MAX_VALUE;
    
    // LLK of the sum of both bins
    double oldCriterion = Double.MAX_VALUE;
    
    // LLK of the sum of both bins
    double newCriterion = Double.MAX_VALUE;
    
    // LLK before/after split difference
    double trainCriterionDiff = Double.MAX_VALUE;
    
    // if true cut and put instances at cutpoint to the right
    boolean rightFlag = false;
    
    // index of the instance where it is split
    int index = -1;
    
    // the middle leftBegin and ends, they differ according to right flag
    int lastLeft = -1;
    int firstRight = -1;
    
    // number of instances to the left of the split
    double leftNum = -1.0;
    
    // number of instances to the right of the split
    double rightNum = -1.0;
    
    // distance of the split towards the left leftEnd
    double leftDist = -1.0;
    
    // distance of the split towards the right leftEnd
    double rightDist = -1.0; 
    
    // value at which the split is performed
    double cutValue = -1.0;
    
    // the bin in which the split happened
    MultiBin bin = null;
    int splitDepth = -1;
    
    public void makeNewLeaveNodesInTree() {
      
      if (bin != null) {
        Tree tree = bin.getTree();
        tree.attrIndex = attrIndex;
        tree.cutValue = cutValue;
        tree.leftNode = new Tree();
        tree.rightNode = new Tree();
      }
    }

  } // leftEnd of class Split

 * */
  
  private void changeIntoDiff(Vector bins, Tree tr_b) {
    double densMax = 0.0;
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin_a = (MultiBin) bins.elementAt(i);
      double dens_a = bin_a.getDensity();
      
      Instance binInst = bin_a.getRepresentative();
      MultiBin bin_b = MultiBinningUtils.findBin(tr_b, binInst);
      double dens_b = bin_b.getDensity();
      bin_a.changeIntoDiff(dens_a, dens_b);
      if (dens_a > densMax) {
	densMax = dens_a;
      }
      if (dens_b > densMax) {
	densMax = dens_b;
      }
    }
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      bin.setDiffMax(densMax);
    }    

    m_tree = m_diffTree;
  }
  
  /*private void changeIntoDiff(Vector bins_a, Vector bins_b) {
    double densMax = 0.0;
    for (int i = 0; i < bins_a.size(); i++) {
      MultiBin bin_a = (MultiBin) bins_a.elementAt(i);
      double dens_a = bin_a.getDensity();
      
      Instance binInst = bin_a.getRepresentative();
      MultiBin bin_b = MultiBinningUtils.findBin(bins_b, binInst);
      double dens_b = bin_b.getDensity();
      bin_a.changeIntoDiff(dens_a, dens_b);
      if (dens_a > densMax) {
	densMax = dens_a;
      }
      if (dens_b > densMax) {
	densMax = dens_b;
      }
    }
    for (int i = 0; i < bins_a.size(); i++) {
      MultiBin bin = (MultiBin) bins_a.elementAt(i);
      bin.setDiffMax(densMax);
    }    
  }*/
  
  private static void changeIntoDiff(Vector bins, Instances data_a, Instances data_b) {
    double densMax = 0.0;
    boolean is_a = true;
 
    for (int i = 0; i < data_a.numInstances(); i++) {
      Instance inst = data_a.instance(i); 
      MultiBinningUtils.addInstanceToBins(bins, inst, 1.0, is_a);
    }
 
    for (int i = 0; i < data_b.numInstances(); i++) {
      Instance inst = data_b.instance(i); 
      MultiBinningUtils.addInstanceToBins(bins, inst, 1.0, !is_a);
    }
    
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      double dens_a = bin.getA_Density();
      double dens_b = bin.getB_Density();
      
       bin.changeIntoDiff(dens_a, dens_b);
      if (dens_a > densMax) {
	densMax = dens_a;
      }
      if (dens_b > densMax) {
	densMax = dens_b;
      }
    }
    
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      bin.setDiffMax(densMax);
    }    
  }
 
  /*public Tree getDiffTree() {
    return m_diffTree;
   }*/
       
  public Tree getTree() {
    return m_tree;
   }
  
   public Vector getBins() {
    return m_bins;
   }
       
    /**
   * Display a representation of this estimator.
   *
   *@return a string giving a representation of the estimator
   */
  public String toString() {
    StringBuffer text = new StringBuffer("MultiDifference: ");
    if (m_bins == null) {
      text = text.append("No bins");
    } else {  
      /*if (dbo.dl(D_FULLRESULTBINS)) {
        text.append(MultiBinningUtils.fullResultsToString(m_bins) +"\n\n");
      } else {
        if (dbo.dl(D_RESULTBINS)) {
          text.append(MultiBinningUtils.binsToString(m_bins) +"\n\n");
        }
      }*/
      if (m_usedAttrs == null) {
	text.append("\nNo Attributes have been cut\n\n");			      
      } else {
	int numAtts = m_usedAttrs.length;
	int cuts = 0;
	for (int i = 0; i < numAtts; i++) {
	  if (m_usedAttrs[i]) cuts++;
	}
	text.append("\nAttributes that have been cut ("+cuts+" of "+numAtts+")\n\n");		
	// wrong!!boolean [] atts = MultiBinningUtils.listOfCuttingAtts(m_bins);
	for (int i = 0; i < numAtts; i++) {
	  if (m_usedAttrs[i]) {
	    String attsName = m_originalData.attribute(i).name();
	    text.append("" + i + ":" + attsName + "\n");		
	  }
	}
	text.append("\n");
      }      
      text.append(MultiBinningUtils.TUBEtoString(m_diffTree, m_bins, "MultiDifference"));
      
    }
    return text.toString();
  }

  /*
   *
   * 
   */
  public double getProbability(Instance inst, double weight) {
    // TODO Auto-generated method stub
    //dbo.pln("getprop "+inst);
    double prob = MultiBinningUtils.getProbability(m_diffTree, inst, weight);
//  if (dbo.dl(D_NULLPROB) && prob == 0.0) DBO.pln("prob is 0.0");
    return prob;
  }
  
  /**
   * @param args
   */
  public static void main(String[] args) {
    
    try {
//    DBO.pln("argument 0 "+argv[0]);
//    DBO.pln("argument 1"+argv[1]);
//    DBO.pln("");     
      //*** MultiTUBENew est = new MultiTUBENew();
    	MultiTUBE est = new MultiTUBE();
      
      MultiEstimator.buildEstimator((MultiEstimator) est, args, false);      
      System.out.println(est.toString());
      
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
  }
    
}
