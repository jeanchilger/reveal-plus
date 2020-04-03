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
 *    MultiBinningUtils.java
 *    Copyright (C) 2009 Gabi Schmidberger
 *
 */
package weka.estimators;

import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Vector;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.Debug.DBO;
import weka.filters.unsupervised.attribute.Bin;

/** 
*
<!-- globalinfo-start -->
* Class containing utility functions for Multi-TUBE.
<!-- globalinfo-leftEnd -->
*
* @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
* @version $Revision: 1.0 $
*/
public class MultiBinningUtils{

  public static class Tree implements Serializable {

    /**
	 * 
	 */
	private static final long serialVersionUID = -3253433563611248830L;

	int attrIndex = -1;

    double cutValue = Double.NaN;

    boolean rightFlag = true;

    Tree leftNode = null;

    Tree rightNode = null;

    MultiBin bin = null;
  }

  public static class GlobalSplitData {

    /** root node to the tree */
    Tree tree;

    /** estimators of all attributes */
    AttrEstimator attrEstimators[];

    /** number to identify prepared splits for the same bin */
    int splitCounter = 0;

    /** number of numeric attributes */
    int numAttr = -1;

    /** class index */
    int classIndex;

    /** used attributes (e.g. not class) */
    boolean[] usedAttr;

    /** the resulting bins */
    Vector bins = new Vector(); //

    /** the valid list */
    boolean[][] valid;

    /** the single attribute datafiles */
    Instances[] data;

    // priority queue handling
    /** the new bins */
    Vector newBins = new Vector(); //

    Vector priorityQueue = new Vector(); //

    double trainCriterionDiff;

    // control data
    /** the max split depth */
    int maxDepth = 0;

    // gathering info about the splitting process
    double numIllegalCuts = 0;

    double avgCVIllegalCuts = 0.0;

    int numTotallyUniform = 0;

    // infos that go into bins
    double bigN = -1.0;

    double[] bigL;

    int[] numInst = null;

    double[] MINValue = null;

    double[] MAXValue = null;

    double[] medianValue = null;

    double[] range = null;

    double numInstForIllCut = 1;

    double alpha = 1.0;

    double totalCriterion;
  }

  /*
   * Class to represent a split.
   */
  public static class Split implements Serializable {

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

  public static class BinPath implements Comparable<BinPath>{
    public String pathString = null;
    public int index = -1;

    BinPath (String str, int i) {
      pathString = str;
      index = i;
    }

    /**
     * Compare method for interface Comparator
     * this is first bin
     * @param b second bin
     * @return -1, 0 or 1 if a < b, == b or > b
     */
    public int compareTo(BinPath b) {
      return pathString.compareTo(b.pathString);
    }
  }

  /**
   * Gathers the bins from a tree
   * 
   * @param tr
   *                the tree to gather the bins from
   */
  public static Vector gatherBins(Tree tr) {
    Vector bins = new Vector();

    gatherBins(tr, bins);
    return bins;
  }

  /**
   * Gathers the bins from a tree
   * 
   * @param tr
   *                the tree to gather the bins from
   */
  private static Vector gatherBins(Tree tr, Vector bins) {

    if (tr != null) {
      gatherBins(tr.leftNode, bins);
      gatherBins(tr.rightNode, bins);

      // is leave node
      if (tr.leftNode == null) {
	// add bin
	bins.add(tr.bin);
      }
    }
    return bins;
  }

  /**
   * Gathers the bins from a tree
   * 
   * @param tr
   *                the tree to gather the bins from
   * @param bins
   *                the bins to gather into
   * @param takeRoot
   *                flag if to take root node
   */
  public static Vector gatherAllBins(Tree tr, Vector bins, boolean takeRoot) {

    // don't add the root node
    if (tr != null) {
      if (takeRoot) {
	bins.add(tr.bin);
      }
      gatherAllBins(tr.leftNode, bins, true);
      gatherAllBins(tr.rightNode, bins, true);
    }
    return bins;
  }

  public static MultiBin[] sortBinsByDensity(Vector bins) {
    MultiBin[] binsList = new MultiBin[bins.size()];
    for (int i = 0; i < bins.size(); i++) {
      binsList[i] = (MultiBin) bins.elementAt(i);
    }
    Arrays.sort(binsList);
    return binsList;
  }

  public static BinPath[] sortBinsByPathString(Vector bins) {
    int numBins = bins.size();
    BinPath[] pathList = new BinPath[numBins];
    for (int i = 0; i < numBins; i++) {
      MultiBin bin =  (MultiBin)bins.elementAt(i);
      pathList[i] = new BinPath(bin.m_splitPath, i);
    }
    Arrays.sort(pathList);
    return pathList;
  }

  /**
   * check if bin is not neighbor of a list of bins
   * @param bin the bin
   * @param index the index up to index -1 the checks should be made
   * @param orderedBinList the list of bins 
   * @return true if bin is not the neighbor of these bins
   */
  public static boolean binNotNeighborOf(MultiBin bin, int index,
      MultiBin[] orderedBinList, Tree tree, int numAttr) {
    boolean hasDenserNeighbor = false;
    int i = 0;
    while (!hasDenserNeighbor && i < index) {
      if (!binNotNeighborOf(bin, orderedBinList[i], tree, numAttr)) {
	hasDenserNeighbor = true;
      }
      i++;
    }
    return !hasDenserNeighbor;
  }

  /*
   * Tests if other bin is neighbor of bin, returns true if NOT
   * It uses the split tree to find the answer. 
   * @param bin the bin the test is made for
   * @param otherBin testing if it is neighbour of bin
   * @param tree the splitting tree
   * @param numAttr stores info about attributes so it need the num of attr
   */
  public static boolean binNotNeighborOf(MultiBin bin,
      MultiBin otherBin, Tree tree, int numAttr) {

    String binPath = bin.getSplitPath();
    String otherBinPath = otherBin.getSplitPath();
    //DBO.pln("p1: "+binPath);
    //DBO.pln("po: "+otherBinPath);
    
    int i = 0;
    boolean notNeighbor = true;
    boolean isIdentical = true;
    int lastCutAttr = -1;

    // ignore first part of path which is identical
    while (i < binPath.length() && i < otherBinPath.length() && isIdentical) {
      // maybe last identical cut
      lastCutAttr = tree.attrIndex;
      // still same path, continue to next part in tree
      if (binPath.charAt(i) == otherBinPath.charAt(i)) {
	if (binPath.charAt(i) == 'L') tree = tree.leftNode;
	else tree = tree.rightNode;
	i++;
      } else {
	// end of identical path 
	isIdentical = false;
      }
    }

    if (lastCutAttr == -1) {
      // are the same bins
      notNeighbor = false;
      return notNeighbor;
    }
    
    // Continue if not the same bins!!
    //--------------------------------
    // initialize attribute flags
    boolean [] cutAttributes = new boolean[numAttr];
    for (int j = 0; j < numAttr; j++) {
      cutAttributes[j] = false;
    }
    cutAttributes[lastCutAttr] = true;

    // gather which attributes are used for cutting
    Tree saveTree = tree;
    int saveI = i;
    // check bin path first bin - first not needed
    if (tree.attrIndex > 0) {
      if (binPath.charAt(i) == 'L') tree = tree.leftNode;
      else tree = tree.rightNode;	  
      i++;
    }
    while (tree.attrIndex > 0) {
      cutAttributes[tree.attrIndex] = true;
      if (binPath.charAt(i) == 'L') tree = tree.leftNode;
      else tree = tree.rightNode;	  
      i++;
    }
    // check bin path second bin - first not needed
    tree = saveTree;
    i = saveI;
    if (tree.attrIndex > 0) {
      if (otherBinPath.charAt(i) == 'L') tree = tree.leftNode;
      else tree = tree.rightNode;
      i++;
    }
    while (tree.attrIndex > 0) {
      cutAttributes[tree.attrIndex] = true;
      if (otherBinPath.charAt(i) == 'L') tree = tree.leftNode;
      else tree = tree.rightNode;	  
      i++;
    }

    // check overlapping of attribute subranges
    boolean allOverlap = true;
    for (int j = 0; j < numAttr && allOverlap; j++) {
      if (cutAttributes[j]) {  	  
	allOverlap =  
	  testOverlap(bin.getMinValue(lastCutAttr), 
	      bin.getMinIncl(lastCutAttr),
	      bin.getMaxValue(lastCutAttr),
	      bin.getMaxIncl(lastCutAttr),
	      otherBin.getMinValue(lastCutAttr), 
	      otherBin.getMinIncl(lastCutAttr),
	      otherBin.getMaxValue(lastCutAttr),
	      otherBin.getMaxIncl(lastCutAttr));
      }
    }
    notNeighbor = !allOverlap;
    //DBO.pln("");
    return notNeighbor;
  }

  /**
   * Tests if the ranges overlap, takes the border into account.
   * Min and max could be part of the range or not (e.g.'(' or '[')
   * @param aMin min of A bin
   * @param aMinIncl is min included of A bin
   * @param aMax max of A bin
   * @param aMaxIncl is max included of A bin
   * @param bMin min of B bin
   * @param bMinIncl is min included of B bin
   * @param bMax max of B bin
   * @param bMaxIncl is max included of B bin
   * @return
   */
  private static boolean testOverlap(double aMin, boolean aMinIncl, double aMax, boolean aMaxIncl, 
      double bMin, boolean bMinIncl, double bMax, boolean bMaxIncl) {
    boolean overlap = true;
    if (bMin >= aMax) {
      overlap = false;
      if (bMin == aMax) {
	// if cut at this point 
	// old error was!!
	//if (bMinIncl || bMaxIncl) overlap = true;
	if (bMinIncl || aMaxIncl) overlap = true;
      }
    } else {
      if (aMin >= bMax) {
	overlap = false;
	if (aMin == bMax) {
	  if (aMinIncl || bMaxIncl) overlap = true;
	}
      }
    }
    return overlap;
  }

  /**
   * Defines an illegal cut.
   * 
   * @param numInst
   *                number of instances
   * @param width
   *                width of the bin
   */
  public static boolean isIllegalCut(int numInst, double width,
      double totalLen, double totalNum) {
    return isIllegalCut((double) numInst, width, totalLen, totalNum);
  }

  /**
   * Returns the threshold for an illegal cut.
   * 
   * @param totalNum
   *                number of all not missing instances
   */
  public static int getIllegalCutThreshhold(double totalNum) {
    double thresh = (((double) totalNum * 0.1) / 100.0) + 1.0;
    int intThresh = (int) (Math.sqrt(thresh));
    // intThresh = 5;
    return intThresh;
  }

  /**
   * Defines an illegal cut.
   * 
   * @param numInst
   *                number of instances
   * @param width
   *                width of the bin
   * @param totalLen
   *                length of total range
   * @param totalNum
   *                number of all not missing instances
   */
  public static boolean isIllegalCut(double numInst, double width,
      double totalLen, double totalNum) {
    int intThresh = getIllegalCutThreshhold(totalNum);
    // Oops.pln("treshhold "+intThresh+" with totalnum "+totalNum);
    if (((width / totalLen) < 0.01))
      //if ((numInst <= intThresh) && ((width / totalLen) < 0.01)) 
      return true;

    return false;

  }

  /**
   * Return the number of almost empty bins (density < 0,1).
   * 
   * @param bins
   *                the bins for the given attribute
   * @return the number of almost empty bins
   */
  public static double getNumAlmEmptyBins(Vector bins, double total)
  throws Exception {

    int num = 0;
    MultiBin bin = null;
    double p;
    // dbo.dp("#");
    for (int j = 0; j < bins.size(); j++) {
      bin = (MultiBin) bins.elementAt(j);
      p = (bin.getWeight() * 100.0) / total;
      if (p <= 1.0) {
	num++;
      }
    }
    // dbo.dpln("");
    return (double) num;
  }

  /**
   * makes a data set with one attribute only copying it from the paameter
   * data set missing can be deleted, result data set can be sorted
   * 
   * @param data
   * @param attrIndex
   * @param sorted
   *                if true return datasets sorted
   * @return
   */
  public static Instances makeOneAttDataset(Instances data, int attrIndex,
      boolean sorted) {

    FastVector attributes = new FastVector(1);
    attributes.addElement(data.attribute(attrIndex).copy());
    Attribute index = new Attribute("index");
    attributes.addElement(index);
    String name = data.relationName() + "ATT" + attrIndex;
    Instances work = new Instances(name, attributes, 0);

    // copy attribute values
    int numInst = data.numInstances();
    double weight = 1.0;

    for (int ii = 0; ii < numInst; ii++) {
      double[] values = new double[2];
      values[0] = data.instance(ii).value(attrIndex);
      values[1] = (double) ii;
      Instance inst = new Instance(weight, values);
      work.add(inst);
    }
    // DBO.pln("makeOneAttDataset"+work);
    if (sorted) {
      work.sort(0);
    }
    return work;
  }

  /**
   * makes a valid list with all valid except the missing one
   * 
   * @param data
   *                the one attribute data set
   * @param deleteMissing
   *                if missing should be set unvalid
   * @return
   */
  public static boolean[] makeFirstValidList(Instances data,
      boolean deleteMissing) {

    int numInst = data.numInstances();
    boolean[] valid = new boolean[numInst];
    int aIndex = 0;

    // find valid values
    for (int ii = 0; ii < numInst; ii++) {
      if (data.instance(ii).isMissing(aIndex) && deleteMissing) {
	valid[ii] = false;
      } else {
	valid[ii] = true;
      }
    }
    return valid;
  }

  /**
   * find the number of missing values in the attribute
   * 
   * @param data
   *                the one-attribute data set
   * @return number of missing values in that attribute
   */
  public static int getNumMissing(Instances data) {

    int numMissing = 0;
    for (int i = 0; i < data.numInstances(); i++) {
      if (data.instance(i).isMissing(0)) {
	numMissing++;
      }
    }
    return numMissing;
  }

  /**
   * Make list of valid flags
   * 
   * @param insts
   *                instances to make valid flags for
   * @return valid flags array
   */
  public static boolean[] makeValidFlags(Instances insts) {
    boolean[] valid = new boolean[insts.numInstances()];
    for (int i = 0; i < valid.length; i++) {
      valid[i] = true;
    }
    return valid;
  }

  /*
   * Compute the density for one cutoff area.
   * 
   * @param num number of instances in the bin @param width width of the
   * bin @param totalNum total number of instances in that area
   */
  protected static double getDensity(double num, double width, double totalNum) {
    if (num == 0.0)
      return 0.0;
    double llh = num / (width * totalNum);
    // double llh = num / (totalNum);
    return llh;
  }

  /*
   * Returns the density of the bin for an estimate of the probability.
   * 
   * @param root the root of the bin tree @param inst the instance @param
   * return the density for an estimate of the probability value
   */
  protected static double getProbability(Vector bins, Instance inst,
      double weight) {
    if (bins == null) {
      return 0.0;
    }
    MultiBin bin = findBin(bins, inst);
    double prob = bin.getDensity();
    return prob * weight;
  }

  /*
   * Returns the density of the bin for an estimate of the probability.
   * 
   * @param root the root of the bin tree @param inst the instance @param
   * return the density for an estimate of the probability value
   */
  protected static double getProbability(Tree root, Instance inst, double weight) {
    if (root == null) {
      return 0.0;
    }
    MultiBin bin = findBin(root, inst);
    double prob = bin.getDensity();
    return prob * weight;
  }

  /**
   * Compute the loglikelihood for one cutoff area.
   * 
   * @param num
   *                number of instances left off the cut point
   * @param width
   *                width of the sub range of the area
   * @param totalNum
   *                number of instances in that area
   */
  protected static double getLoglikelihood(double num, double width,
      double totalNum) {
    // dbo.dpln("getLoglikelihood num "+num+" width "+width+" totalNum
    // "+totalNum);
    if (num == 0)
      return 0.0;
    double density = num / (width * totalNum);
    double llk = (num) * (Math.log(density));

    // dbo.dpln("Loglikelihood "+llk);
    return llk;
  }

  /**
   * Use the given bins and fill the instances into the given bins.
   * Returns the average loglikelihood, which is the sum of loglikelihoods
   * divided by the number of instances.
   * 
   * @param bins
   *                the bins for the given attribute
   * @param test
   *                a set of instances, need not be the one the bins have
   *                been build for
   * @param index
   *                the index of the attribute the loglikelihood is asked
   *                for
   */
  public static double getLoglkFromBins(Tree root, Instances test) {
    double loglk = 0.0;

    // DBO.pln("getloglk ***");
    for (int i = 0; i < test.numInstances(); i++) {
      Instance inst = test.instance(i);
      // DBO.pln("getloglk "+ inst);
      // find bin
      Tree node = root;
      while (!Double.isNaN(node.cutValue)) {
	double value = inst.value(node.attrIndex);
	if (value < node.cutValue) {
	  node = node.leftNode;
	} else {
	  if (value > node.cutValue) {
	    node = node.rightNode;
	  } else {
	    if (value == node.cutValue) {
	      if (node.rightFlag) {
		node = node.rightNode;
	      } else {
		node = node.leftNode;
	      }
	    }
	  }
	}
      }

      loglk += node.bin.getOneLoglk();
    }
    return loglk;
  }

  /**
   * Empty the bins by removing test instances (not the weights = training
   * instances).
   * 
   * @param bins the list of bins
   */
  public static void emptyBins(Vector bins) {
    // Oops.pln("emptyBins");
    if ((bins == null) || (bins.size() == 0))
      return;
    for (int j = 0; j < bins.size(); j++) {
      MultiBin bin = (MultiBin) bins.elementAt(j);
      bin.emptyBin();
      bin.emptyB_Bin();
    }
  }

  /**
   * Empty the bins by removing B instances (not the weights = training
   * instances).
   * @param bins the list of bins
   */
  public static void emptyB_Bins(Vector bins) {
    // Oops.pln("emptyB_Bins");
    if ((bins == null) || (bins.size() == 0))
      return;
    for (int j = 0; j < bins.size(); j++) {
      MultiBin bin = (MultiBin) bins.elementAt(j);
      bin.emptyB_Bin();
    }
  }

  /**
   * Empty the bins by removing TRAIN instances (set the weights to zero)
   * @param bins the list of bins
   */
  public static void emptyBinsAsTrain(Vector bins) {
    // Oops.pln("emptyBins");
    if ((bins == null) || (bins.size() == 0))
      return;
    int numBins = bins.size();
    for (int j = 0; j < numBins; j++) {
      MultiBin bin = (MultiBin) bins.elementAt(j);
      bin.emptyBinAsTrain();
    }
  }

  /**
   * Refills the bins using the weight of the instances and
   * treating the instances as test instances
   * @param data the instances used to refill
   * @param bins the bins to fill into
   * @param checkClass flag if negative and positive should be differed
   */
  public static void refillBins(Instances data, Vector bins, boolean checkClass) {
    MultiBinningUtils.emptyBins(bins);
    boolean isA = true;
    double totalNum_A = 0;
    double totalNum_B = 0;
    for (int i = 0; i < data.numInstances(); i++) {
      Instance inst = data.instance(i); 
      double instWeight = inst.weight();
      if (checkClass) {
	double cl = inst.classValue();
	if (cl == 0.0) {
	  isA = true; totalNum_A += instWeight;
	} else {
	  isA = false; totalNum_B += instWeight;
	}
      }
      //DBO.p("instance: "+i+" weight "+instWeight);
      MultiBinningUtils.addInstanceToBins(bins, inst, instWeight, isA);
    }
    if (checkClass) {
      for (int j = 0; j < bins.size(); j++) {
	MultiBin bin = (MultiBin) bins.elementAt(j);
	bin.setA_BtotalNums(totalNum_A, totalNum_B);
	//bin.setTotalNum(totalNum_A + totalNum_B);
      }      
    }
  }

  /**
   * Refills as train; refills the bins using the weight of the instances and
   * treating the instances as TRAIN instances (weight is added to weight in bin)
   * @param data the instances used to refill
   * @param bins the bins to fill into
   */
  public static void refillBinsAsTrain(Instances data, Vector bins) {
    MultiBinningUtils.emptyBinsAsTrain(bins);
    double totalNum = 0;
    for (int i = 0; i < data.numInstances(); i++) {
      Instance inst = data.instance(i); 
      double instWeight = inst.weight();
      totalNum+= instWeight;
      // DBO.p("instance: "+i+" weight "+instWeight);
      MultiBinningUtils.addInstanceToBinsAsTrain(bins, inst, instWeight);
    }
    for (int j = 0; j < bins.size(); j++) {
      MultiBin bin = (MultiBin) bins.elementAt(j);
      bin.setTotalNum(totalNum);
      //bin.setTotalNum(totalNum_A + totalNum_B);
    }      
  }

  /**
   * Use the given bins and fill the instances into the given bins.
   * Returns the average loglikelihood, which is the sum of loglikelihoods
   * divided by the number of instances.
   * 
   * @param bins
   *                the bins for the given attribute
   * @param test
   *                a set of instances, need not be the one the bins have
   *                been build for
   * @param index
   *                the index of the attribute the loglikelihood is asked
   *                for
   */
  public static MultiBin findBin(Tree root, Instance inst) {
    MultiBin bin = null;

    // find bin
    Tree node = root;
    double cutValue = node.cutValue;
    while (!Double.isNaN(cutValue)) {
      double value = inst.value(node.attrIndex);
      if (value < cutValue) {
	node = node.leftNode;
	cutValue = node.cutValue;
      } else {
	if (value > cutValue) {
	  node = node.rightNode;
	  cutValue = node.cutValue;
	} else {
	  if (value == cutValue) {
	    if (node.rightFlag) {
	      node = node.rightNode;
	      cutValue = node.cutValue;
	    } else {
	      node = node.leftNode;
	      cutValue = node.cutValue;
	    }
	  }
	}
      }
    }
    bin = node.bin;
    return bin;
  }


  public static MultiBin findBin(Vector bins, Instance inst) {
    MultiBin bin = null;
    // MultiBin lastResultBin = null;
    // boolean notFound = true;
    for (int i = 0; i < bins.size(); i++) {
      bin = (MultiBin) bins.elementAt(i);
      if (bin.fitsInto(inst)) {
	return bin;
	// lastResultBin = bin;
      }
    }
    return null;
    // return lastResultBin;
  }

  /**
   * Finds if instance falls into any of the bins disallowing the border values,
   * returns null if not
   * 
   * @param bins vector of bins to check
   * @param inst instance that is tested
   * @return bin if it falls into one, null if not
   */
  public static MultiBin findInsideOfBin(Vector bins, Instance inst) {
    MultiBin bin = null;
    // MultiBin lastResultBin = null;
    // boolean notFound = true;
    for (int i = 0; i < bins.size(); i++) {
      bin = (MultiBin) bins.elementAt(i);
      if (bin.fitsInside(inst)) {
	return bin;
	// lastResultBin = bin;
      }
    }
    return null;
    // return lastResultBin;
  }

  /**
   * Add weight of one instance to the bin. Enter as A instance or as B
   * instance.
   * 
   * @param bin the bin to enter the instance
   * @param weight the weight of the instance
   * @param enterasA true if to enter as A, false if to enter as B
   */
  public static void addInstToBin(MultiBin bin, double weight, boolean enterasA) {
    if (enterasA) {
      bin.addInst(weight);
    } else {
      bin.addB_Inst(weight);
    }
  }

  /**
   * Fills all bins allong the path of the instance with the weight given.
   * Enter as A instance or as B instance.
   * 
   * @param root
   *                the root node of the bin tree
   * @param inst
   *                the instance to be entered
   * @param the
   *                weight to enter the instance with
   * @param enterasA
   *                true if to enter as A, false if to enter as B
   */
  public static MultiBin addInstanceToTree(Tree root, Instance inst,
      double weight, boolean enterasA) {
    MultiBin bin = null;

    // find bin
    Tree node = root;
    double cutValue = node.cutValue;

    // while not leave node found
    while (!Double.isNaN(cutValue)) {

      bin = node.bin;
      addInstToBin(bin, weight, enterasA);
      // ***DBO.pln("add into: "+bin.dimensionsToString());

      double value = inst.value(node.attrIndex);
      if (value < cutValue) {
	node = node.leftNode;
	cutValue = node.cutValue;
      } else {
	if (value > cutValue) {
	  node = node.rightNode;
	  cutValue = node.cutValue;
	} else {
	  if (value == cutValue) {
	    if (node.rightFlag) {
	      node = node.rightNode;
	      cutValue = node.cutValue;
	    } else {
	      node = node.leftNode;
	      cutValue = node.cutValue;
	    }
	  }
	}
      }
    }
    bin = node.bin;
    // ***DBO.pln(""+bin);
    addInstToBin(bin, weight, enterasA);
    return bin;
  }

  public static void addInstanceToTree(Tree root, Instance inst, double weight)
  throws Exception {
    MultiBin bin = findBin(root, inst);
    if (bin != null) {
      bin.addInst(weight);
    } else {
      throw new Exception("Bin not found; for instance \n" + inst);
    }
  }

  /**
   * Adds one test instance to the bins.
   * @param bins the bins to fill in
   * @param inst the instance added
   * @param weight weight of the instance
   * @param isA flag if instance is negative (A) or positive (B)
   * @return
   */
  public static MultiBin addInstanceToBins(Vector bins, Instance inst,
      double weight, boolean isA) {
    //DBO.pln("addInstanceToBins");
    MultiBin bin = null;
    MultiBin lastResultBin = null;
    boolean didFit = false;
    // boolean notFound = true;
    int numBins = bins.size();
    //DBO.pln("Instance \n"+inst+" numBins "+numBins);

    for (int i = 0; i < numBins && !didFit; i++) {
      //if (i == 2) {
	//DBO.pln("Does fit??? - bin "+i);
      //}
      bin = (MultiBin) bins.elementAt(i);
      if (bin.fitsInto(inst)) {
	didFit = true;
	//DBO.pln("Does fit - bin "+i);
	if (isA)
	  bin.addInst(weight);
	else
	  bin.addB_Inst(weight);
	lastResultBin = bin;
      } 
      //else {
	//DBO.pln("Doesn't fit - bin "+i);
	//DBO.pln(bin.fullBinToString());
      //}
    }
    if (!didFit) 
      DBO.pln("Didn't fit any bin "+inst);
     
    //DBO.pln("");
    return lastResultBin;
  }

  /**
   * Adds one TRAIN instance to the first bin it fits into
   * @param bins the bins to fill in
   * @param inst the instance added
   * @param weight weight of the instance
   * @return the bin it fitted into
   */
  public static MultiBin addInstanceToBinsAsTrain(Vector bins, Instance inst,
      double weight) {
    MultiBin bin = null;
    MultiBin resultBin = null;
    boolean didFit = false;
    // boolean notFound = true;
    int numBins = bins.size();
    //DBO.pln("Instance \n"+inst+" numBins "+numBins);

    for (int i = 0; i < numBins && !didFit; i++) {
      bin = (MultiBin) bins.elementAt(i);
      if (bin.fitsInto(inst)) {
	didFit = true;
	//DBO.pln("Does fit - bin "+i);
	bin.addInstAsTrain(weight);
	resultBin = bin;
      } 
      /*else {
	DBO.pln("Doesn't fit - bin "+i);
	DBO.pln(bin.fullBinToString());
      }*/
    }
    //DBO.pln("");
    return resultBin;
  }

  public static double[] getBinFilling(Vector bins, double classValue) {

    double[] attr = new double[bins.size() + 1];
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      attr[i] = bin.getNumInst();
    }
    attr[attr.length - 1] = classValue;
    return attr;
  }

  /**
   * Returns list of attributes with value percentage of the B instances
   * in the bins
   * 
   * @param bins
   *                the vector with the bins
   * @param classValue
   *                the classvalue to be added as last attribute
   * @return
   */
  public static double[] getBinB_PercentFilling(Vector bins, double classValue) {
    double[] attr = new double[bins.size() + 1];
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      attr[i] = bin.getB_Percent();
    }
    attr[attr.length - 1] = classValue;
    return attr;
  }

  /**
   * Returns list of attributes with value percentage of the B instances
   * in the bins
   * 
   * @param bins
   *                the vector with the bins
   * @param classValue
   *                the classvalue to be added as last attribute
   * @param numInstInBag
   *                the number of instances in the bag
   * @return
   */
  public static double[] getBinB_PercentFilling(Vector bins, double classValue,
      double numInstInBag) {
    double[] attr = new double[bins.size() + 1];
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      attr[i] = bin.getB_Percent(numInstInBag);
    }
    attr[attr.length - 1] = classValue;
    return attr;
  }

  /**
   * Returns list of attributes with value percentage of the B instances
   * in the bins
   * 
   * @param bins
   *                the vector with the bins
   * @param classValue
   *                the classvalue to be added as last attribute
   * @return
   */
  public static double[] getBinAofAllPercentFilling(Vector bins,
      double classValue) {
    double[] attr = new double[bins.size() + 1];
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      attr[i] = bin.getAofAllPercent();
    }
    attr[attr.length - 1] = classValue;
    return attr;
  }

  /**
   * Returns list of attributes with value percentage of the B instances
   * in the bins
   * 
   * @param bins
   *                the vector with the bins
   * @param classValue
   *                the classvalue to be added as last attribute
   * @return
   */
  public static double[] getBinBofAllPercentFilling(Vector bins,
      double classValue) {
    double[] attr = new double[bins.size() + 1];
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      attr[i] = bin.getBofAllPercent();
    }
    attr[attr.length - 1] = classValue;
    return attr;
  }

  /**
   * Returns list of attributes with difference of A to B instances (num_A -
   * num_B)
   * 
   * @param bins
   *                the vector with the bins
   * @param classValue
   *                the classvalue to be added as last attribute
   * @return
   */
  public static double[] getBinDiffABFilling(Vector bins, double classValue) {
    double[] attr = new double[bins.size() + 1];
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      attr[i] = bin.getDiffAB();
    }
    attr[attr.length - 1] = classValue;
    return attr;
  }

  /**
   * Transforms the bins to cutinfo.
   * 
   * @param bins
   *                the vecor of bins
   * @return the list of cutpoints
   */
  public static MultiCutInfo binsToCutInfo(Vector bins) {

    // no cutpoints here
    if (bins == null)
      return null;

    // transform bins into cutpoints
    MultiCutInfo info = new MultiCutInfo(bins.size() - 1);

    for (int i = 0; i < info.numCutPoints(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      int attr = bin.getLastCutAttr();
      info.m_cutAttr[i] = attr;
      info.m_cutPoints[i] = bin.getMaxValue(attr);
      info.m_cutAndLeft[i] = bin.getMaxIncl(attr);
    }
    return info;
  }

  /**
   * Returns the bins of one dimensional view
   * 
   * @param attrIndex
   *                the index of the attribute to get the bins from
   * @param multiBins
   *                the multidimensional bins
   * @return
   */
  public static Vector getAttrBins(int attrIndex, Vector multiBins) {
    Vector bins = null;

    return bins;
  }

  public static boolean[] splitFromValid(boolean[] valid, double cutValue,
      boolean rightFlag, int attrIndex, boolean left, Instances[] data,
      Instances globalData) {
    // DBO.pln("splitFromValid");

    // attributes index always 0
    int aIndex = 0;
    // index index always 1
    int iIndex = 1;

    int numInst = data[attrIndex].numInstances();
    boolean flag = true;
    if (!left) {
      flag = false;
    }
    boolean beginFlag = flag;

    // split it up into 4 cases
    for (int ii = 0; ii < numInst; ii++) {
      int index = (int) data[attrIndex].instance(ii).value(iIndex);
      if (!data[attrIndex].instance(ii).isMissing(aIndex)) {
	double value = data[attrIndex].instance(ii).value(aIndex);
	if (beginFlag == flag) {
	  if (rightFlag) {
	    if (data[attrIndex].instance(ii).value(aIndex) >= cutValue) {
	      flag = !flag; // DBO.pln(" switch");

	    }
	  }
	  if (!rightFlag) {
	    if (data[attrIndex].instance(ii).value(aIndex) > cutValue) {
	      flag = !flag; // DBO.pln(" switch");
	    }
	  }
	}
	// DBO.p(" "+ value); if (!valid[index]) {DBO.p("#");} else
	// {DBO.p("o");}
	// DBO.p(" "+index+"="+ globalData.instance(index)); if (!valid[index])
	// {DBO.p("#");} else {DBO.p("o");}
      }
      valid[index] = valid[index] && flag;
      // if (!valid[index]) {DBO.pln("#");} else {DBO.pln("o");}
    }
    // DBO.pln("");
    // could be made faster by knowing leftBegin and leftEnd index of bin
    return valid;
  }

  /**
   * Returns a string representing the bins
   */
  public static String fullBinsToString(Vector bins) {
    int numIllegalCuts = 0;
    StringBuffer text = new StringBuffer("\n");
    if (bins == null)
      text.append("# no bins\n");
    else {
      for (int i = 0; i < bins.size(); i++) {
	MultiBin bin = (MultiBin) bins.elementAt(i);
	// count number of illegal cuts
	bin.setIllegalCut();
	if (bin.getIllegalCut()) {
	  numIllegalCuts++;
	}

	text.append("#" + i + ": " + bin.fullBinToString());
      }
    }
    // text.append("# "+numIllegalCuts+" illegal cuts.\n");
    return text.toString();
  }

  /**
   * Returns a string representing the bins
   */
  public static String fullBinsToPictBlock(Vector bins, boolean showDiff) {
    int numIllegalCuts = 0;
    StringBuffer text = new StringBuffer("\n");
    if (bins == null)
      text.append("# no bins\n");
    else {
      for (int i = 0; i < bins.size(); i++) {
	MultiBin bin = (MultiBin) bins.elementAt(i);
	// count number of illegal cuts
	bin.setIllegalCut();
	if (bin.getIllegalCut()) {
	  numIllegalCuts++;
	}

	text.append("#" + i + ": ");
	text.append(bin.fullBinToPictBlock(showDiff));
      }
    }
    // text.append("# "+numIllegalCuts+" illegal cuts.\n");
    return text.toString();
  }

  /**
   * Returns a string representing the bins with the text and attribute
   * name for each cut off area
   * 
   * @param bins
   *                the bins of the discretization done
   * @param the
   *                data model that supports the partition
   * @return the text
   */

  public static String fullBinsToRulesText(Vector bins, Instances dataModel) {
    int numIllegalCuts = 0;
    StringBuffer text = new StringBuffer("\n");
    if (bins == null)
      text.append("# no bins\n");
    else {
      for (int i = 0; i < bins.size(); i++) {
	MultiBin bin = (MultiBin) bins.elementAt(i);
	text.append("#" + i + ": ");
	text.append(bin.fullBinToRulesText(dataModel));
      }
    }
    return text.toString();
  }

  /**
   * Returns a string representing the bins
   */
  public static String fullResultsToString(Vector bins) {
    int numIllegalCuts = 0;
    StringBuffer text = new StringBuffer("\n");
    if (bins == null)
      text.append("# no bins\n");
    else {
      for (int i = 0; i < bins.size(); i++) {
	MultiBin bin = (MultiBin) bins.elementAt(i);
	// count number of illegal cuts
	bin.setIllegalCut();
	if (bin.getIllegalCut()) {
	  numIllegalCuts++;
	}

	text.append("#" + i + ": " + bin.fullResultsToString());
      }
    }
    // text.append("# "+numIllegalCuts+" illegal cuts.\n");
    return text.toString();
  }

  /**
   * Returns a string representing the bins
   */
  public static String binsToString(Vector bins) {
    int numIllegalCuts = 0;
    StringBuffer text = new StringBuffer(
	"#|Att|Bin                    |    %    ||  Instances |   %     | Volume | Density | Probability |\n"
	+ "#+-------------------------------------------------------------+-----------------------+-------------------+\n");
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      // count number of illegal cuts
      bin.setIllegalCut();
      if (bin.getIllegalCut()) {
	numIllegalCuts++;
      }

      text.append("#" + i + ": " + bin.toString());
    }
    text.append("# " + numIllegalCuts + " illegal cuts.\n");
    return text.toString();
  }

  public static String binsToNumString(Vector bins) {
    StringBuffer text = new StringBuffer("");
    int numBins = bins.size(); 
    for (int i = 0; i < numBins; i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      text.append("#" + i + ": " + bin.toNumString());
    }
    return text.toString();
  }

  /**
   * Returns a string representing the bins
   */
  public static String diffBinsToString(Vector bins) {
    int numIllegalCuts = 0;
    StringBuffer text = new StringBuffer(
	"#|Att|Bin                    |    %    || Diffdensity  \n"
	+ "#+---------------------------------------------------------------------------------------------------------+\n");
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      // count number of illegal cuts
      bin.setIllegalCut();
      if (bin.getIllegalCut()) {
	numIllegalCuts++;
      }

      text.append("#" + i + ": " + bin.toDiffString());
    }
    text.append("# " + numIllegalCuts + " illegal cuts.\n");
    return text.toString();
  }

  /**
   * Returns a string with the short version of the difference bins
   * representation
   */
  public static String diffBinsToABString(Vector bins, boolean ab,
      boolean allAttr) {
    StringBuffer text = new StringBuffer("\n");
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);

      text.append("#" + i + ": " + bin.toABString(ab, allAttr));
    }
    return text.toString();
  }

  /**
   * Returns a string with the short version of the difference bins
   * representation
   */
  public static String binsToAttrOnlyCutString(Vector bins, Instances data,
      int totalLength, int afterComma) {
    StringBuffer text = new StringBuffer("\n");
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      text.append("#" + i + ": \n"
	  + bin.getAllAttrOnlyCutString(totalLength, afterComma, data));

    }
    return text.toString();
  }

 /**
  *
  * Returns a string with the short version of the difference bins
  * representation.
  * @param bins
  * @param ab
  * @param shDensity
  * @param shWeight
  * @param shVolume
  * @param oneLine
  * @param maxDensity
  * @param maxABDensity
  * @param secondLine
  * @return
  */
  public static String binsToPictStringRow(Vector bins, boolean ab,
      boolean shDensity, boolean shWeight,
      boolean shVolume, boolean oneLine, double maxDensity, double maxABDensity, boolean secondLine) {
    StringBuffer text = new StringBuffer("\n");
    double totalNum = ((MultiBin) bins.elementAt(0)).getTotalNum(); // num all instances
    double num = 0.0; // num of represented instances
    int numBins = bins.size();
    for (int i = 0; i < numBins; i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);

      // if weight show on diagram than add it
      //double w = bin.getWeight();
      //double p = (w * 100.0) / totalNum;
      //if (p >= 0.1) { num += w; }
      double numA = bin.getNumInst();
      double numB = bin.getNumB_Inst();
      double w = numA + numB; //bin.getWeight();
      double p = (w * 100.0) / totalNum;
      if (p >= 0.1) { num += w; }

      text.append("#" + toIntNumString(i, 4) + ": ");

      if (oneLine)
	text.append("");
      else
	text.append("\n");

      text.append(bin.toPictStringRow(ab, shDensity,
	  shWeight, shVolume, true, maxDensity, maxABDensity));
      String pa = bin.getSplitPath();
      double dd = bin.getDivDensity();
      text.append(" inst "+Utils.doubleToString(p, 5, 2)+"% " + pa + " DD "+Utils.doubleToString(dd, 5, 2));
      text.append(" Na "+Utils.doubleToString(numA, 5, 2)+ " Nb " +Utils.doubleToString(numB, 5, 2)+ "");
      text.append(" wgt " + Utils.doubleToString(w, 5, 2));
      text.append("\n");
      if (secondLine)
	text.append( bin.toNiceString()+ "\n");

      //if (dbo.dl(D_BINS)) { //
      //text.append("fullbintostring"+bin.fullBinToString()+"\n");

    }
    double per = percent(totalNum, num);
    text.append("Percentage of instances presented: "+Utils.doubleToString(per, 2)+"%\n");
    return text.toString();
  }

  public static double percent(double total, double sub) {
    double p = 0.0;

    if (sub > 0.0) {
      p = (sub * 100.0) / total;
    }
    return p;
  }

  /**
   * Writes data to file that can be used to plot a histogram.
   * Filename is aprameter f + ".hist".
   *
   *@param f string to build filename
   *@param bins vector of bins
   */

  public static void writeGnuplotBins(String f, Vector bins,
				    double factor) throws Exception {

    PrintWriter output = null;
    Bin bin = null;
    StringBuffer text = new StringBuffer("");
    
    if (f.length() != 0) {
      // add extension to filename
      String name = f + ".hist";
      output = new PrintWriter(new FileOutputStream(name));
    } else {
      return;
    }
    
    if (bins == null) return;
    if (bins.size() == 0) return;
    // first bin
    bin = (Bin) bins.elementAt(0);
    try {
      text.append("" + bin.getMinValue()+" "+0.0+" \n");
      
      for (int i = 0; i < bins.size(); i++) {
	double density = bin.getDensity() * factor;
	//Oops.pln("i "+i+" density "+density);
	text.append(""+bin.getMinValue()+" "+density+" \n");
	text.append(""+bin.getMaxValue()+" "+density+" \n");
	if (i + 1 < bins.size()) {
	  bin = (Bin) bins.elementAt(i + 1);
	}
      }
      text.append("" + bin.getMaxValue()+" "+0.0+" \n");
      // last bin
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
    output.println(text.toString());    

    // close output
    if (output != null) {
      output.close();
    }
  }

  /**
   * Makes a string for the rectangle that the bin makes in the two given
   * dimensions
   *
   *@param bin a bin to make output for
   *@param xDim the attribute that is on the x dimension (horizontal)
   *@param yDim the attribute that is on the y dimension (vertical)
   *@return a string to produce output for gnuplot
   */

  public static String writeGnuplotOneBin(MultiBin bin, double xDim, double yDim) 
  throws Exception {

    StringBuffer text = new StringBuffer("");
    
    /* 
        for (int i = 0; i < bins.size(); i++) {
	double density = bin.getDensity() * factor;
	//Oops.pln("i "+i+" density "+density);
	text.append(""+bin.getMinValue()+" "+density+" \n");
	text.append(""+bin.getMaxValue()+" "+density+" \n");
	if (i + 1 < bins.size()) {
	  bin = (Bin) bins.elementAt(i + 1);
	}
      
      text.append("" + bin.getMaxValue()+" "+0.0+" \n");*/
      // last bin
  /*  } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
    output.println(text.toString());    

    // close output
    if (output != null) {
      output.close();
    }*/
      return text.toString();
  }

 /**
   * Get the maximum value of density over all bins
   * 
   * @param bins
   *                the bins to get the max density from
   * @return the maximum density found over all bins
   */
  public static double getMaxDensity(Vector bins) {
    double maxDensity = 0.0;
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      double dens = bin.getDensity();
      if (dens > maxDensity)
	maxDensity = dens;
    }
    return maxDensity;
  }

  /**
   * Get the maximum value of density over all bins
   * 
   * @param bins
   *                the bins to get the max density from
   * @return the maximum density found over all bins
   */
  public static double getMaxABDensity(Vector bins) {
    double maxDensity = 0.0;
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      double dens = bin.getA_Density();
      if (dens > maxDensity)
	maxDensity = dens;
      dens = bin.getB_Density();
      if (dens > maxDensity)
	maxDensity = dens;

    }
    return maxDensity;
  }
  
  public static boolean acceptByDensity(Vector bins, int min, double maxDensity) {
    boolean isAccepted = false;
    int numDensHigher = 0;
    for (int i = 0; i < bins.size(); i++) {
      MultiBin bin = (MultiBin) bins.elementAt(i);
      int p = percentNextInt(maxDensity, bin.getDensity());
      if (p > 1)
	numDensHigher++;
    }
    if (numDensHigher > min)
      isAccepted = true;
    return isAccepted;
  }

  /**
   * Returns the percentage in integers 1 for 0%-10%, 2 for 10%-20%,....
   * 
   * @param total
   *                the total value
   * @param sub
   *                the part of the value that should be transformed to %
   * @return the integer representing the parts of 10 of the percentage
   */
  protected static int percentNextInt(double total, double sub) {
    int max = 0;
    double p = 0.0;

    if (sub > 0.0) {
      if (sub == total) {
	max = 10;
      } else {
	p = (sub * 100.0) / total;
	max = (int) Math.rint(0.5 + (p / 10.0));
      }
    }
    return max;
  }

  /**
   * Returns the percentage as a string
   * 
   * @param total
   *                the total value
   * @param sub
   *                the part of the value that should be transformed to %
   * @param len
   *                the maximal length of the string
   * @return the string representing the percentage value
   */
  protected static String toIntNumString(int num, int len) {
    String str = Utils.doubleToString((double) num, 0)
    + "                                   ";
    return str.substring(0, len - 1);
  }

  /**
   * Build a representationstring of the estimator of these bins.
   * 
   */
  public static String simpleTUBEtoString(Vector bins, String name) {
    double sum = 0.0;
    StringBuffer result = new StringBuffer("" + name + " Estimator. Counts = ");
    if (bins != null) {
      for (int i = 0; i < bins.size(); i++) {
	MultiBin bin = (MultiBin) bins.elementAt(i);
	double num = bin.getWeight();
	sum += num;
	result.append(" " + Utils.doubleToString(num, 2));
      }
      result.append("  (Total = " + Utils.doubleToString(sum, 2) + " / "
	  + bins.size() + " bins).\n");
      return result.toString();
    } else {
      result.append(" (No Bins).\n");
    }
    return result.toString();
  }

  /**
   * Build a representationstring of the estimator of these bins.
   * 
   */
  public static String TUBEtoString(Tree root, Vector bins, String name) {
    double sum = 0.0;
    StringBuffer result = new StringBuffer("" + name + " Estimator. Counts = ");
    if (bins != null) {
      for (int i = 0; i < bins.size(); i++) {
	MultiBin bin = (MultiBin) bins.elementAt(i);
	double num = bin.getWeight();
	sum += num;
	result.append(" " + Utils.doubleToString(num, 2));
      }
      result.append("  (Total = " + Utils.doubleToString(sum, 2) + " / "
	  + bins.size() + " bins).\n");
      result.append("\nVolumes ");
      for (int i = 0; i < bins.size(); i++) {
	MultiBin bin = (MultiBin) bins.elementAt(i);
	double num = bin.getVolume();
	sum += num;
	result.append(" " + Utils.doubleToString(num, 2));
      }
      result.append("\n\nDensities ");
      for (int i = 0; i < bins.size(); i++) {
	MultiBin bin = (MultiBin) bins.elementAt(i);
	double num = bin.getDensity();
	sum += num;
	result.append(" " + Utils.doubleToString(num, 2));
      }
      return result.toString();
    } else {
      result.append(" (No Bins).\n");
    }
    if (root != null) {

    } else {
      StringBuffer treeString = new StringBuffer("");
      result.append(treeToString(root, treeString));

    }
    return result.toString();
  }

  public static String treeToString(Tree root, StringBuffer result) {
    Tree node = root;
    result.append("" + node.attrIndex);
    if (node.leftNode != null)
      treeToString(node.leftNode, result.append(" ("));
    if (node.rightNode != null) {
      treeToString(node.rightNode, result.append(" "));
      result.append(")");
    }
    return result.toString();
  }

  /**
   * @param args
   */
  public static void main(String[] args) {
    // TODO Auto-generated method stub

  }

}
