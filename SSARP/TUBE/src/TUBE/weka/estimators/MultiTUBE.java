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
 *    MultiTUBE.java
 *    Copyright (C) 2009 Gabi Schmidberger
 *
 */
package weka.estimators;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.Debug.DBO;
import weka.estimators.MultiBinningUtils.GlobalSplitData;
import weka.estimators.MultiBinningUtils.Split;
import weka.estimators.MultiBinningUtils.Tree;

/** 
*
<!-- globalinfo-start -->
* Class for multidimensional discretizing algorithm using TUBE discretization.
<!-- globalinfo-leftEnd -->
*
* @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
* @version $Revision: 1.0 $
*/
public class MultiTUBE extends MultiBinningEstimator {

  /**
	 * 
	 */
	private static final long serialVersionUID = -8929054405753116908L;

/** illegal cut as it happens */
  public static int D_ILLCUT          = 5; // 6 

  /** num cuts and its averaged entropy */
  public static int D_NUMCUTENTROPY   = 6; // 7

  /** print priority queue info */
  public static int D_PRIORITY        = 8; // 9

  /** list all cutpoints at leftEnd */
  public static int D_SUMILLCUTS      = 12; // 13

  /** list all cutpoints at leftEnd */
  public static int D_NUMCUTMISE      = 13; // 14

  /** trace through precedures */
  public static int D_TRACE           = 14; // 15

  /** print info from all bins not only centres */
  public static int D_WIDEPICT        = 16;

  /** look at at each real bin made */
  public static int D_LOOKATBINS      = 18; // 19

  /** probability is zero */
  public static int D_NULLPROB        = 19; // 20

  /** list all cuts made */    
  public static int D_LISTCUTS        = 20; // 21

  /** list all new bins made */    
  public static int D_NEWBINBOXES     = 21; // 22

  private static final int DENSITY = 1;
  private static final int DIVDENSITY = 2;

  /** holds the choice of the splitting methods */
  private int m_compareMethod = DENSITY;

  /** the epsilon taken to cut beside a value  */
  private double m_epsilon = 1.0E-4;

  /** the default value of epsilon  */
  private double m_defaultEpsilon = 1.0E-4;

  /** original instances */
  protected Instances m_originalData = null;

  /** model estimator, will be copied for every numerical attribute */
  protected AttrEstimator m_modelNumEstimator = new AttrTUBEPure();

  /** estimators of all attributes */
  protected AttrEstimator m_attrEstimators[];

  /** number of attributes */
  protected int m_numAttr;

  /** The maximum splitting depth */
  protected int m_maxDepth = Integer.MAX_VALUE;

  /** The maximum number of splits */
  protected int m_maxSplits = 99;

  /** The current maximum number of splits */
  protected int m_currentMaxSplits = -1;

  /** Store the current cutpoints in the order they are done */
  protected boolean m_storeSplits = false;

  /** list of all used attributes */
  protected boolean[] m_usedAttrs = null;

  /** vector to store the cuts */
  protected Vector m_splitList = new Vector();

  //stuff set by options

  /** different splitting criteria */
  private static final int CV_SPLIT = 1;
  private static final int FULL_SPLIT = 2;
  private static final int NOEMPTY_SPLIT = 5;

  /** holds the choice of the splitting methods */
  private int m_splitMethod = FULL_SPLIT;

  /** The seed used for discretization */
  protected int m_seed;

  /** flag, if set crossvalidation for number of splits is switched off*/
  private boolean m_noCVNumSplits = false;

  /** flag, if cuts are set on the grid */
  private boolean m_gridCutting = false;

  /** number of grid cells */
  private int m_gridNum = -1;

  /** Dont allow illegal cuts. */
  protected boolean m_forbidIllegalCut = true;

  /** list all cutpoints at leftEnd */
  protected static final int C_TOOHIGH_EW10      = 1;

  /** Default for forbiddenCut. */
  private static final int M_FORBIDDENCUT = -1;

  /** Dont allow different types of cuts. */
  protected int m_forbiddenCut = M_FORBIDDENCUT;
  //leftEnd-of-stuff set by options

  /** the centre bins */
  private Vector m_centreBins = null;

  /** the cluster bin lists */
  private Vector [] m_clusterBinList = null;

  protected class InfoModule {
    int numTotallyUniform = 0;
    int numAlmEmptyBins = 0;   
    double numIllegalCuts = 0.0;
    double avgCVIllegalCuts= 0;
    double error = 0.0; 
    int minNumber = -1;
    double minLlk = Double.MAX_VALUE;
  }

  /**
   * Returns default capabilities of the clusterer.
   *
   * @return      the capabilities of this clusterer
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // attributes
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.NOMINAL_CLASS);
    //result.enable(Capability.DATE_ATTRIBUTES);
    
    // other
    result.setMinimumNumberInstances(2);
    return result;
  }

  /**
   * Initialize the estimator with a set of values.
   * @param whatever just to differ it
   * @param data the dataset used to build this estimator 
   */
  public void addValues(boolean whatever, Instances data) throws Exception {

    dbo.dpln(D_TRACE, "addValues \n");
    InfoModule doku = new InfoModule();

    int numFolds = 10;
    m_originalData = data;
    Instances workData = new Instances(data);

    GlobalSplitData spGlobal = initializeGlobalInfo(data, m_maxDepth,
	m_numInstForIllCut, m_alpha);
    m_tree = spGlobal.tree;

    try {
    	Random random = new Random(m_seed);

    	// cross validation over number of splits
    	int bestNumber = -1;
    	if (!m_noCVNumSplits &&  (spGlobal.bigN > numFolds)) {
    		boolean stopCV = false;
    		int cvNumInstances = workData.numInstances() * (numFolds - 1) / numFolds;

    		// zero split entropy 
    		double llk = 0.0;

    		//double uniformLlk = 0.0;
    		double trainLlk = 0.0;
    		double maxLlk = - Double.MAX_VALUE;
    		double numBins = 0.0; 

    		m_currentMaxSplits = -1;
    		Instances [] trainSet = new Instances[numFolds];
    		Instances [] testSet = new Instances[numFolds];
    		GlobalSplitData [] spGlobals = new GlobalSplitData[numFolds];

    		workData.randomize(random);

    		// get all train and testsets
    		for (int i = 0; i < 10; i++) {
    			trainSet[i] = workData.trainCV(numFolds, i, random);
    			testSet[i] = workData.testCV(numFolds, i);

    			// prepare splitting information
    			spGlobals[i] = initializeGlobalInfo(trainSet[i], m_maxDepth,
    					m_numInstForIllCut, m_alpha);   
    		}
    		double [] aveDensity = null;

    		// ****************************************************************************
    		// cross validation over number of splits
    		do {
    			m_currentMaxSplits++;
    			llk = 0.0; trainLlk = 0.0; numBins = 0.0; 
    			doku.numIllegalCuts = 0.0;
    			doku.numTotallyUniform = 0;
    			doku.numAlmEmptyBins = 0;
    			doku.error = 0.0;
    			for (int i = 0; i < numFolds; i++) {
    				//if (i == 5) {
    				//DBO.pln("\n"+trainSet[i]);
    				//DBO.pln("# fold "+i+" maxsplits "+ m_currentMaxSplits);
//  				EstimatorUtils.binsToString(spInfos[i].bins));
    				//}

    				// Build the discretization using the train fold
    				trainLlk += performSplits(spGlobals[i]);

    				numBins += spGlobals[i].bins.size();
    				aveDensity = new double[spGlobals[i].bins.size()];
    				doku.numIllegalCuts += (double)spGlobals[i].numIllegalCuts;
    				doku.numTotallyUniform += (double)spGlobals[i].numTotallyUniform;

    				double n = MultiBinningUtils.getNumAlmEmptyBins(spGlobals[i].bins, trainSet[i].numInstances());
    				doku.numAlmEmptyBins += n;

    				// test discretization
    				double l = 0.0;

    				l = MultiBinningUtils.getLoglkFromBins(spGlobals[i].tree, testSet[i]);

//  				if (dbo.dl(D_NUMCUTENTROPY)) { 
//  				dbo.dpln(EstimatorUtils.binsToString(spInfos[i].bins));
//  				dbo.dpln("#loglikeli "+i+" = "+l);
//  				}

    				llk += l;

    				//setAveDensity(aveDensity, spGlobals[i].bins);
    				//doku.error += getErrorsFromBins(spGlobals[i].bins);
    			}

    			// set averages
    			llk = llk / numFolds;


    			// -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    			// compute further statistics
    			trainLlk = trainLlk / numFolds;
    			numBins = numBins / numFolds;
    			doku.numIllegalCuts = doku.numIllegalCuts / numFolds;
    			doku.numTotallyUniform = doku.numTotallyUniform  / numFolds;
    			doku.numAlmEmptyBins = doku.numAlmEmptyBins / numFolds;
    			doku.error = doku.error / numFolds;
    			if (dbo.dl(D_NUMCUTENTROPY)) { 
    				dbo.dpln("" + m_currentMaxSplits + "  " + llk + 
    						" trainLlk "+trainLlk+" err "+doku.error+" aeB " + doku.numAlmEmptyBins + " bins " 
    						+ numBins + " illegalcuts "+ doku.numIllegalCuts +
    						" totallyuniform "+ doku.numTotallyUniform);
//  				dbo.dpln("" + m_currentMaxSplits + "  " + llk +
//  				" trainLlk "+trainLlk+" err "+error+" aeB " + numAlmEmptyBins + " bins " 
//  				+ numBins + " illegalcuts "+ numIllegalCuts +
//  				" totallyuniform "+ numTotallyUniform);

    			}

    			// found new maximum
    			if (llk > maxLlk) {
    				maxLlk = llk;
    				bestNumber = m_currentMaxSplits;
    				doku.avgCVIllegalCuts = doku.numIllegalCuts;
    			}

    			// -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    			// output min and max
    			if (llk < doku.minLlk) {
    				doku.minLlk = llk;
    				doku.minNumber = m_currentMaxSplits;
    			}
    			if (dbo.dl(D_NUMCUTENTROPY)) {
    				dbo.dpln("#max "+bestNumber+" min "+doku.minNumber);
    			}
    			// -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

    			stopCV = false;
    			if  ((m_maxSplits > -1) && (m_currentMaxSplits >= m_maxSplits)) { 
    				stopCV = true; 
    			}
    			//if (m_currentMaxSplits > 100)  { stopCV = true; }
    			if (m_currentMaxSplits > cvNumInstances)  { stopCV = true; }
    		} while (!stopCV);
    		m_currentMaxSplits = bestNumber;
    	}
    	// leftEnd cross-validation
    	//dbo.dpln("leftEnd cross-validation");     

    	// set maximum number of splits, might have been computed by cross-validation
    	if (m_currentMaxSplits == -1) {
    		if (m_maxSplits != -1) {
    			m_currentMaxSplits =  m_maxSplits;
    		} else {
    			m_currentMaxSplits =  workData.numInstances();
    		}
    	}

    	// perform splits on full data
    	m_splitList = new Vector();

    	m_storeSplits = true;
    	performSplits(spGlobal);

    	// memorize average number of illegal cuts of CV plus actual
    	if (!m_noCVNumSplits) {
    		m_avgCVIllegalCuts = (double)doku.avgCVIllegalCuts;
    		m_numIllegalCuts = (double)spGlobal.numIllegalCuts;
    		m_diffNumIllegalCuts = (double)doku.avgCVIllegalCuts - (double)spGlobal.numIllegalCuts;
    	} else {
    		m_avgCVIllegalCuts = -1.0;
    		m_numIllegalCuts = (double)spGlobal.numIllegalCuts;
    		m_diffNumIllegalCuts = -1.0;
    	}

    	// -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    	if (dbo.dl(D_SUMILLCUTS)) {
    		// output number of illegal cuts
    		dbo.dpln("#Average of Illegal cuts in CV: "+ doku.avgCVIllegalCuts);
    		dbo.dpln("#Illegal cuts:                : "+ spGlobal.numIllegalCuts);
    		m_diffNumIllegalCuts = doku.avgCVIllegalCuts - (double)spGlobal.numIllegalCuts;
    		dbo.dpln("#Difference                   : "+ m_diffNumIllegalCuts);
    	}
    	// -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

    	// fill global bins
    	m_bins = spGlobal.bins;

    	// -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    	if (dbo.dl(D_FULLRESULTBINS)) {
    		dbo.dpln(MultiBinningUtils.fullResultsToString(m_bins) +"\n\n");
    	} else {
    		if (dbo.dl(D_RESULTBINS)) {
    			// output cutpoints
    			MultiCutInfo info = getCutInfo();
    			if (info.m_cutPoints == null) {
    				dbo.dpln("\n# no cutpoints found  " ); 
    			} else {
    				//dbo.dpln("\n#* "+info.m_cutPoints.length+" cutpoint(s) ");

//  				for (int i = 0; i < info.m_cutPoints.length; i++) {
//  				dbo.dp("# "+info.m_cutPoints[i]+" "); 
//  				dbo.dpln(""+info.m_cutAndLeft[i]);
//  				}
    				//for (int i = 0; i < m_splitList.size(); i++) {
    				//  Split split = (Split)  m_splitList.elementAt(i);
    				//  dbo.dpln("# "+"|" + split.attrIndex+"|" + split.cutValue+
    				//      "|"+split.bin.fullToString());
    				//}
    				//dbo.dpln("# leftEnd");
    				dbo.dpln(MultiBinningUtils.binsToString(m_bins));
    			}
    		}
    	}
    	if (dbo.dl(D_WIDEPICT)) {
    		// output cutpoints
    		MultiCutInfo info = getCutInfo();
    		if (info.m_cutPoints == null) {
    			dbo.dpln("\n# no cutpoints found  " ); 
    		} else {

    			double maxDensity = MultiBinningUtils.getMaxDensity(m_bins);

    			dbo.dpln(MultiBinningUtils.binsToPictStringRow(m_bins, false, true, true, 
    					true, true, maxDensity, 0.0, false));
    			// -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    		}
    	}
    } catch (Exception ex) {
    	ex.printStackTrace();
    	System.out.println(ex.getMessage());
    }   
    }

  /*
   * Initialize the global data.
   * @param data the data set, used for data model information
   * @return the global information for the splits
   */
  protected GlobalSplitData initializeGlobalInfo(Instances data, int maxDepth,
      double numInstForIllCut, double alpha) throws Exception {
    /** estimators of all attributes */

    GlobalSplitData spInfo = new GlobalSplitData();

    // set options for model estimator
    setEstimatorOptions(m_modelNumEstimator);//

    m_numAttr = data.numAttributes();
    // make all estimators
    spInfo.attrEstimators = 
      (AttrEstimator [])AttrEstimator.makeCopies(m_modelNumEstimator, data.numAttributes());

    // initialize estimators with the data 
    for (int i = 0; i < m_numAttr; i++) {
      if (i != m_classIndex && i != getNotUsedAttribute())
	spInfo.attrEstimators[i].initializeNewData(data, i, m_MINValue[i], m_MAXValue[i]);
    }
    spInfo.data = new Instances[m_numAttr];

    spInfo.bigN = data.numInstances();
    spInfo.numAttr = m_numAttr;

    spInfo.maxDepth = maxDepth;
    spInfo.numInstForIllCut = numInstForIllCut;
    spInfo.alpha = alpha;

    spInfo.usedAttr = new boolean[m_numAttr];
    spInfo.bigL = new double[m_numAttr];
    spInfo.numInst = new int[m_numAttr];
    spInfo.MINValue = m_MINValue;
    spInfo.MAXValue = m_MAXValue;   
    spInfo.range = m_range;
    spInfo.valid = new boolean[m_numAttr][];
    spInfo.classIndex = m_classIndex;

    for (int i = 0; i < m_numAttr; i++) {
      spInfo.usedAttr[i] = true;
      if (i == m_classIndex) spInfo.usedAttr[i] = false;
      if (i == m_notUsedAttribute) spInfo.usedAttr[i] = false;
      spInfo.data[i] = spInfo.attrEstimators[i].getData();
      spInfo.bigL[i] = spInfo.attrEstimators[i].getTotalLen();
      spInfo.numInst[i] = spInfo.attrEstimators[i].getNumInst();
      //spInfo.MINValue[i] = spInfo.attrEstimators[i].getMINValue();
      //spInfo.MAXValue[i] = spInfo.attrEstimators[i].getMAXValue();
      spInfo.valid[i] = spInfo.attrEstimators[i].getValid();
    }

    // make root node
    spInfo.tree = new Tree();

    // start with one bin for the whole range
    MultiBin bin = new MultiBin(spInfo, m_originalData, m_compareMethod);
    spInfo.bins.add(bin);
    spInfo.newBins.add(bin);
    return spInfo;
  }

  /**
   * Perfom the splits.
   * @param spInfo holds all the relevant 'global' values
   * @return the total entropy of the result split
   */
  protected double performSplits(GlobalSplitData spInfo) throws Exception {

    boolean maxSplitReached = false;
    double diff = 0.0;
    boolean [] usedAttrs = null;
    if (m_storeSplits) {
      usedAttrs = new boolean[spInfo.numAttr];
      // store cutpoints in order made
      for (int i = 0; i < spInfo.numAttr; i++) {
	usedAttrs[i] = false;
      }
    }
    // number of splits already larger or equal than allowed splits
    if (spInfo.bins.size() - 1 >= m_currentMaxSplits) {
      maxSplitReached = true;
    } else {
      //
      // split until no more new splits found or stopping criterion reached
      do {
	// look through all the new bins to find new possible splits
	for (int i = 0; i < spInfo.newBins.size(); i++) {
	  MultiBin bin = (MultiBin)spInfo.newBins.elementAt(i);

	  Split split = new Split();
	  split.trainCriterionDiff = -Double.MAX_VALUE;

	  // go through all attributes of the bin to find new split
	  spInfo.splitCounter++;
	  for (int attrIndex = 0; attrIndex < m_numAttr; attrIndex++) {
	    if (attrIndex != spInfo.classIndex && attrIndex != getNotUsedAttribute()) {

	      if (m_MAXValue[attrIndex] > m_MINValue[attrIndex]) {
		Split newSplit = ((AttrBinningEstimator)spInfo.attrEstimators[attrIndex]).findOneSplit(spInfo, bin);

		if (newSplit != null) {
		  // split was found
		  if (newSplit.trainCriterionDiff > split.trainCriterionDiff) {
		    split = newSplit;
		  }
		}
	      }
	    }
	  }

	  // if one split found take the best among all attributes
	  if (split.trainCriterionDiff > -Double.MAX_VALUE) {
	    addToPriorityQueue(spInfo.priorityQueue, split);
	  }
	} // leftEnd of looking through new bins

	// all new bins have been examined
	spInfo.newBins = new Vector();

	// dbo.dpln("spInfo.priorityQueue.size() "+spInfo.priorityQueue.size());
	// dbo.dpln("maxSplit "+maxSplit);

	// get next split 
	if ((spInfo.priorityQueue.size() > 0) && !maxSplitReached) {

	  // output priority queue - -- -- -- -- -- -- -- -- -- -- -- -- -- 
	  if (dbo.dl(D_PRIORITY)) {
	    dbo.dpln("# ");
	    for (int j = 0; j < spInfo.priorityQueue.size(); j++) {
	      Split s = (Split)spInfo.priorityQueue.elementAt(j);
	      dbo.dp("#:" + spInfo.priorityQueue.size() + "::---");
	      double total = spInfo.totalCriterion + (s.trainCriterionDiff / spInfo.bigN);
	      dbo.dpln("# "+total+"/_"+s.trainCriterionDiff+"/_"+s.cutValue);
	    }
	    dbo.dpln("# ");
	  } // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

	  // take first from priority queue and perform the split
	  Split split = (Split)spInfo.priorityQueue.elementAt(0);
	  spInfo.priorityQueue.remove(0);
	  removeFromPriorityQueue(spInfo.priorityQueue, split.splitNumber);
	  if (dbo.dl(D_PRIORITY)) {
	    dbo.dpln("#::---"+split.cutValue+" taken.");
	  }   

	  // enter tree info into bins
	  split.makeNewLeaveNodesInTree();

	  // make two new bins, means now really perform next split
	  boolean left = true;
	  split.bin.getTree().attrIndex = split.attrIndex;
	  split.bin.getTree().cutValue = split.cutValue;

	  //public static boolean [] splitFromValid(double cutValue, boolean rightFlag, 
	  //  int attrIndex, boolean left, boolean[][] fullValid, Instances [] data) {
	  //DBO.pln("split at " + split.cutValue +" with attr "+ split.attrIndex);
	  MultiBin leftBin = new MultiBin(split.bin, spInfo, split,
	      m_numInstForIllCut, m_alpha, left, m_originalData);
	  //
	  MultiBin rightBin = new MultiBin(split.bin, spInfo, split,
	      m_numInstForIllCut, m_alpha, !left, m_originalData);
	  //DBO.pln("right\n"+rightBin.fullResultsToString());

	  // output bin boxes -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	  if (dbo.dl(D_NEWBINBOXES)) {
	    dbo.dpln("#left   :\n"+leftBin.rangesToString(false)); 
	    dbo.dpln("#right  :\n"+rightBin.rangesToString(false)); 
	    dbo.dpln("# ");
	  } // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

	  // notify attribute estimator about the split
	  AttrBinningEstimator est = (AttrBinningEstimator)spInfo.attrEstimators[split.attrIndex];
	  est.setSplit(split.cutValue);

	  // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	  if (m_storeSplits) {
	    // store cutpoints in order made
	    m_splitList.add(split);
	    usedAttrs[split.attrIndex] = true;
	  }
	  // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

	  // test if illegal cut was chosen
	  leftBin.setIllegalCut();
	  if (leftBin.getIllegalCut()) {
	    spInfo.numIllegalCuts++;
	  }
	  rightBin.setIllegalCut();
	  if (rightBin.getIllegalCut()) {
	    spInfo.numIllegalCuts++;
	  }

	  // after split, two new bins, that will have to be examined
	  spInfo.newBins.add(leftBin);
	  spInfo.newBins.add(rightBin);
	  // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	  if (dbo.dl(D_LOOKATBINS)) {
	    dbo.dpln("#Binleft\n" + leftBin.toString());
	    dbo.dpln("#Binright\n" + rightBin.toString());         
	  }
	  if (dbo.dl(D_LISTCUTS)) {
	    dbo.dpln("#cut at attr "+leftBin.getLastCutAttr()+" : "+
		split.cutValue + "  " +
		leftBin.getWeight() + "/" + rightBin.getWeight());
	    if ((leftBin.getWeight() != leftBin.m_testWeight) ||
		(rightBin.getWeight() != rightBin.m_testWeight)) {
	      dbo.dpln("weight mismatch!!!");
	    }
	  }
	  // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

	  // new total llk
	  spInfo.totalCriterion = spInfo.totalCriterion 
	  + (split.trainCriterionDiff / spInfo.bigN);

	  // delete the old bin in bins and put in the two new ones
	  int binIndex = 0;
	  for (int i = 0; i < spInfo.bins.size(); i++) {
	    if (split.bin == (MultiBin)spInfo.bins.elementAt(i)) {
	      binIndex = i;
	    }
	  }
	  spInfo.bins.remove(binIndex);
	  spInfo.bins.addAll(binIndex, spInfo.newBins);
	  // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	  // output total entropy 
	  if (dbo.dl(D_PRIORITY)) {
	    //dbo.dpln("#Loglk = "+  getTrainLoglkFromBins(spInfo.bins, spInfo.bigN));
	  }

	  // leftEnd if max splits have been done
	  if ((spInfo.bins.size() - 1) >= m_currentMaxSplits) {
	    maxSplitReached = true;
	  }
	}

      } while (((spInfo.priorityQueue.size() > 0) || (spInfo.newBins.size() > 0))
	  && (!maxSplitReached));
    }

    if (m_storeSplits) {
      m_usedAttrs = usedAttrs;
    }

    // perform splits on bins
    double loglk =  spInfo.totalCriterion;
    return loglk / spInfo.bigN;
  }

  /**
   * Add to priority queue at the position of priority.
   * @param priorityQueue the priority queue
   * @param newSplit the new split element
   */
  protected void addToPriorityQueue(Vector priorityQueue, Split newSplit) {

    for (int i = 0; i < priorityQueue.size(); i++) {
      Split split = (Split)priorityQueue.elementAt(i);
      if (newSplit.trainCriterionDiff > split.trainCriterionDiff) {
	priorityQueue.add(i, newSplit);

//	dbo.dp("/"+i+"/");
//	for (int j = 0; j < priorityQueue.size(); j++) {
//	Split s = (Split)priorityQueue.elementAt(j);
//	dbo.dpln("cut@  "+s.cutValue+" entropy "+s.totalEntropy+" totEntValue "+s.entropy);
//	dbo.dp("///");

//	}
//	dbo.dpln("");
	return;
      }
    }
    //    dbo.dpln("#:"+priorityQueue.size()+":split added :last "+newSplit.cutValue);
    priorityQueue.add(newSplit);
  }

  /**
   * Add to priority queue at the position of priority.
   * @param priorityQueue the priority queue
   * @param num the number of which splits have to be deleted
   */
  protected void removeFromPriorityQueue(Vector priorityQueue, int num) {

    for (int i = 0; i < priorityQueue.size(); i++) {
      Split split = (Split)priorityQueue.elementAt(i);
      if (split.splitNumber == num) {
	priorityQueue.remove(i);
	i--;      
      }
    }
  }

  /**
   * finds one split
   * @param spInfo global data used for the split
   * @param bin the bin to find the split in
   * @param attrIndex the attributes index
   * @return the details of the split
   */
  protected Split findSplit(GlobalSplitData spInfo, MultiBin bin, int attrIndex) 
  throws Exception {

    Split split = ((AttrBinningEstimator)spInfo.attrEstimators[attrIndex]).findOneSplit(spInfo, bin);
    return split;
  }  

  /**
   * Gets the value for the random seed
   *
   * @return the setting of the seed
   */
  public int getSeed() {
    return m_seed;
  }

  /**
   * Sets the value for the random seed
   * @param seed the new seed value
   */
  public void setSeed(int seed) {
    m_seed = seed;
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
   * Gets true if decision on split is not done by cross validating
   * @return the setting of the cross validation flag
   */
  public boolean getNoCVNumSplits() {
    return m_noCVNumSplits;
  }

  /**
   * Sets the flag, if set to true cross validating is switched off
   * @param newFlag the new flag value
   */
  public void setNoCVNumSplits(boolean newFlag) {
    m_noCVNumSplits = newFlag;
    if (newFlag)
      setFullSplitting(true);
  }

  /**
   * Set splitting method to full splitting.
   * @param full boolean if true method is set to full splitting
   */
  public void setFullSplitting(boolean full) {
    if (full) {
      m_splitMethod = FULL_SPLIT;
    }
  }

  /**
   * Returns true if splitting method is set to full splitting
   * @return true if splitting method is currently set to full splitting
   */
  public boolean getFullSplitting() {
    return (m_splitMethod == FULL_SPLIT);
  }

  /**
   * Set splitting method to 'no empty bins allowed' splitting.
   * @param f boolean if true method is set to no-empty splitting
   */
  public void setNoEmptySplitting(boolean f) {
    if (f) {
      m_splitMethod = NOEMPTY_SPLIT;
    } 
  }

  /**
   * Returns true if splitting method is set to 'no empty bins allowed'.
   * @return boolean, if true method is currently set to no-empty splitting,
   */
  public boolean getNoEmptySplitting() {
    return (m_splitMethod == NOEMPTY_SPLIT);
  }

  /**
   * Set splitting method to full splitting.
   * @param full boolean if true method is set to full splitting
   */
  public void setGridCutting(boolean newFlag) {
    m_gridCutting = newFlag;
    if (m_gridCutting)
      m_modelNumEstimator = new AttrTUBEGrid();
  }

  /**
   * Returns true if splitting method is set to grid cutting
   * @return true if splitting method is currently set to grid cutting
   */
  public boolean getGridCutting() {
    return m_gridCutting;
  }

  /**
   * Set number of grid cells for grid cutting
   * @param full int number of gridcells
   */
  public void setGridNum(int num) {
    m_gridNum = num; 
    setGridCutting(true);
  }

  /**
   * Returns number of grid cells for grid cutting 
   * @return number of grid cells for grid cutting
   */
  public int getGridNum() {
    return m_gridNum;
  }

  /** 
   * Sets which kind of illegal cuts should be disallowed
   * 
   * @param num new mode value
   */
  public void setForbiddenCut(int num) {
    m_forbiddenCut = num;
  }

  /** 
   * Sets whether illegal cuts are dissallowed
   * @return true if illegal cuts are set to be disallowed
   */
  public int getForbiddenCut() {
    return m_forbiddenCut;
  }

  /** 
   * Sets which kind of compare method sshould be used
   * @param num new compare method
   */
  public void setCompareMethod(int num) {
    m_compareMethod = num;
  }

  /** 
   * Sets whether illegal cuts are dissallowed
   * @return true if illegal cuts are set to be disallowed
   */
  public int getCompareMethod() {
    return m_compareMethod;
  }

  /** 
   * Sets whether illegal cuts (<10% of length and <= 2 instances) should be
   * dissallowed
   * @param flag new flag value
   */
  public void setForbidIllegalCut(boolean flag) {
    m_forbidIllegalCut = flag;
  }

  /** 
   * Sets whether illegal cuts are dissallowed
   * @return true if illegal cuts are set to be disallowed
   */
  public boolean getForbidIllegalCut() {
    return m_forbidIllegalCut;
  }

  /**
   * Sets the maximum number of bins the range can be split into
   *
   * @param max the maximum number of splits
   */
  public void setMaxNumBins(int max) {
    m_maxSplits = max - 1;
  }

  /**
   * Returns the maximum number of bins 
   *
   * @return max the maximum number of binss
   */
  public int getMaxNumBins() {
    return m_maxSplits + 1;
  }

  /**
   * Gets the current settings of the filter.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getEstimatorOptions() {

    String [] options = new String [17];
    int current = 0;

    if (getAlpha() > 0) {
      options[current++] = "-A"; options[current++] = ""+getAlpha();
    }
    if (getGridCutting()) {
      options[current++] = "-G"; options[current++] = ""+getGridNum();
    }
    options[current++] = "-Z"; options[current++] = ""+getEpsilon();
    if (getNoCVNumSplits()) {
      options[current++] = "-N";
    }
//  if (getSplitClass()) {
//  options[current++] = "-C";
//  }
    if (getFullSplitting()) {
      options[current++] = "-F";
    }
    if (getNoEmptySplitting()) {
      options[current++] = "-Y";
    }
    if (getForbidIllegalCut()) {
      options[current++] = "-L";
    }
    if (getForbiddenCut() != M_FORBIDDENCUT) {
      options[current++] = "-U" + getForbiddenCut();
    }

    while (current < options.length) {
      options[current++] = "";
    }
    return options; 
  }

  /**
   * Gets an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(2);

    newVector.addElement(new Option(
	"\tDisallow illegal cuts.",
	"L", 0, "-L"));
    newVector.addElement(new Option(
	"\tDisallow cuts that make a bin higher the 2x the EW 10 "+
	"bins maximum height (\"crude height control\").",
	"U", 0, "-U"));
    newVector.addElement(new Option(
	"\tChoose full splitting (no penalty).",
	"F", 0, "-F"));
    newVector.addElement(new Option(
	"\tChoose grid cutting.",
	"G", 0, "-G"));
    return newVector.elements();
  }

  /**
   * Hands the options given further to the estimator
   * @param est the estimator
   * @exception Exception if an option is not supported
   */
  public void setEstimatorOptions(AttrEstimator est) throws Exception {
    String[] options = getEstimatorOptions();
    est.setOptions(options);
  }

  /**
   * Parses the options for this object. Valid options are: <p>
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    m_splitMethod = FULL_SPLIT;
    super.setOptions(options); 

    // set seed for random
    String seedString = Utils.getOption('S', options);
    if (seedString.length() > 0)
      setSeed(Integer.parseInt(seedString));

    setNoCVNumSplits(Utils.getFlag('N', options));
    if (getNoCVNumSplits()) { m_splitMethod = FULL_SPLIT; }
   
    setFullSplitting(Utils.getFlag('F', options));

    String gridNum = Utils.getOption('G', options);
    if (gridNum.length() > 0) {
      setGridNum(Integer.parseInt(gridNum));
    }

    setForbidIllegalCut(Utils.getFlag('L', options));

    String fCuts = Utils.getOption('U', options);  
    if (fCuts.length() != 0) {
      setForbiddenCut(Integer.parseInt(fCuts));
    } 

    String comparemethod = Utils.getOption('Y', options);  
    if (comparemethod.length() != 0) {
      setCompareMethod(Integer.parseInt(comparemethod));
    } 

    String epsilonString = Utils.getOption('Z', options);
    if (epsilonString.length() > 0) {
      setEpsilon(Double.parseDouble(epsilonString));
    }

    String numBins = Utils.getOption('B', options);
    if (numBins.length() != 0) {
      setMaxNumBins(Integer.parseInt(numBins));
    }
  }

  /**
   * Gets the current settings of the filter.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {

    String [] options = new String [17];
    int current = 0;

    if (getMaxNumBins()> 0) {
      options[current++] = "-B"; options[current++] = ""+getMaxNumBins();
    }
    if (getAlpha() > 0) {
      options[current++] = "-A"; options[current++] = ""+getAlpha();
    }
    if (getNotUsedAttribute() > 0) {
      options[current++] = "-I"; options[current++] = ""+getNotUsedAttribute();
    }
    if (getNoCVNumSplits()) {
      options[current++] = "-N";
    }
//  if (getSplitClass()) {
//  options[current++] = "-C";
//  }
    if (getFullSplitting()) {
      options[current++] = "-F";
    }
    if (getNoEmptySplitting()) {
      options[current++] = "-Y";
    }
    if (getForbidIllegalCut()) {
      options[current++] = "-L";
    }
    if (getForbiddenCut() != M_FORBIDDENCUT) {
      options[current++] = "-U " + getForbiddenCut();
    }

    if (getEpsilon() != m_defaultEpsilon) {
      options[current++] = "-Z " + getEpsilon();
    }

    if (getGridCutting()) {
      options[current++] = "-G" + getGridNum();
    }
    while (current < options.length) {
      options[current++] = "";
    }
    return options; 
  }

  public boolean[] getUsedAttrs() {
    return m_usedAttrs;
  }
  
  /**
   * Display a representation of this estimator.
   *
   *@return a string giving a representation of the estimator
   */
  public String toString() {
    StringBuffer text = new StringBuffer("MultiTUBE estimator: ");
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
	text.append("\nAttributes that have been cut\n\n");		
	// wrong!!boolean [] atts = MultiBinningUtils.listOfCuttingAtts(m_bins);
	int numAtts = m_usedAttrs.length;
	for (int i = 0; i < numAtts; i++) {
	  if (m_usedAttrs[i]) {
	    String attsName = m_originalData.attribute(i).name();
	    text.append("" + i + ":" + attsName + "\n");		
	  }
	}
	text.append("\n");
      }
      text.append(MultiBinningUtils.TUBEtoString(m_tree, m_bins, "MultiTUBE"));
      if (dbo.dl(D_NUMCUTENTROPY) || dbo.dl(D_NUMCUTMISE)) {
	return "# "+text;
      }
    }
    return text.toString();
  }

  @Override
  public double getProbability(Instance inst, double weight) {
    // TODO Auto-generated method stub
    //dbo.pln("getprop "+inst);
    double prob = MultiBinningUtils.getProbability(m_tree, inst, weight);
    if (dbo.dl(D_NULLPROB) && prob == 0.0) DBO.pln("prob is 0.0");
    return prob;
  }

  /**
   * Get all bins that are high density centres. That means all their surronding 
   * centres have lower density
   * @return the vector of bins that are centres
   */
  public void findCentreBins() {
    MultiBin [] orderedBinList = MultiBinningUtils.sortBinsByDensity(m_bins);
    Vector centres = new Vector();
    centres.add(orderedBinList[0]);
    for (int i = 1; i < orderedBinList.length; i++) {
      if (MultiBinningUtils.binNotNeighborOf(orderedBinList[i], i, orderedBinList, m_tree, 
	  m_originalData.numAttributes())) {
	centres.add(orderedBinList[i]);	
      }
    }
  }

  public Vector getCentreBins() {
    return m_centreBins;
  }

  public Vector [] getClusterBinList() {
    return m_clusterBinList;
  }

  /**
   * Get all bins that are high density centres. That means all their surronding 
   * centres have lower density
   * @return the vector of bins that are centres
   */
  public void findCentreBins(boolean useClusterBinList) {

    Vector centres = new Vector();
    if (!useClusterBinList) {
      findCentreBins();
      return;
    }
    MultiBin [] orderedBinList = MultiBinningUtils.sortBinsByDensity(m_bins);
    //for (int i = 0; i < orderedBinList.length; i++) {
    //DBO.pln(""+orderedBinList[i].getDensity());
    //}
    //DBO.pln("DivDensity");
    //for (int i = 0; i < orderedBinList.length; i++) {
    //  DBO.pln(""+orderedBinList[i].getDivDensity());
    //}


    int numBins = orderedBinList.length;
    int [] belongsToCentre = new int[numBins];
    int [] isANeighborOf = new int[numBins];
    int [] isCentreNum = new int[numBins];
    for (int i = 0; i < numBins; i++) {
      belongsToCentre[i] = -1;
      isANeighborOf[i] = -1;
      isCentreNum[i] = -1;
    }

    // most dens bin is centre
    centres.add(orderedBinList[0]);
    isCentreNum[0] = 0;
    belongsToCentre[0] = 0;

    // find all neighbor and neighbor-neighbor bins of one centre after another
    // start with densest centre
    Vector clusterBinList = new Vector();
    int thisCentre = 0;
    int nextCentre = -1;
    while (thisCentre >= 0) {
      Vector binList = new Vector();
      for (int i = thisCentre + 1; i < orderedBinList.length; i++) {
	boolean hasDenserNeighbor = false;
	// test if bin i has a denser neighbor among the denser bins
	for (int j = thisCentre; j < i && !hasDenserNeighbor; j++) {
	  if (!MultiBinningUtils.binNotNeighborOf(orderedBinList[i], orderedBinList[j], m_tree,
	      m_originalData.numAttributes())) {
	    // DBO.pln("Bin "+i+"is neighbor of "+j);
	    hasDenserNeighbor = true;
	    belongsToCentre[i] = belongsToCentre[j];
	    isANeighborOf[i] = j;
	  }
	    //DBO.pln("Bin "+i+"is NOT neighbor of "+j);
	}
	// if didn't have denser neighbor
	if (!hasDenserNeighbor) {
	  if (nextCentre < 0) nextCentre = i;

	  isCentreNum[i] = centres.size();
	  //if (thisCentre == 0)
	  centres.add(orderedBinList[i]);

	  belongsToCentre[i] = i;
	  isANeighborOf[i] = i;
	}
      }

      // gather info in lists
      for (int i = 0; i < numBins; i++) {
	if (belongsToCentre[i] == thisCentre)
	  binList.add(orderedBinList[i]);
      }
      clusterBinList.add(binList);

      // prepare for next centre
      thisCentre = nextCentre;
      nextCentre = -1;
      if (thisCentre > 0) {
	for (int i = thisCentre + 1; i < numBins; i++) {
	  belongsToCentre[i] = -1;
	  isANeighborOf[i] = -1;
	  isCentreNum[i] = -1;
	}      
      }
    } // leftEnd while

    int numCentres = centres.size();
    m_centreBins = centres;
    m_clusterBinList = new Vector[numCentres];
    for (int i = 0; i < numCentres; i++) {
      m_clusterBinList[i] = (Vector)clusterBinList.elementAt(i);
    }
  }

  /**
   * Get all bins that are high density centres. That means all their surronding 
   * centres have lower density
   * @return the vector of bins that are centres
   */
  public void findOnlyNonZeroCentreBins(boolean useClusterBinList) {

    //DBO.pln("findOnlyNonZeroCentreBins");
    Vector centres = new Vector();
    if (!useClusterBinList) {
      findCentreBins();
      return;
    }

    // sort bins, bins know compare method
    MultiBin [] orderedBinList = MultiBinningUtils.sortBinsByDensity(m_bins);
    //for (int i = 0; i < orderedBinList.length; i++) {
    //DBO.pln(""+orderedBinList[i].getDensity());
    //}
    //DBO.pln("DivDensity");
    //for (int i = 0; i < orderedBinList.length; i++) {
    //  DBO.pln(""+orderedBinList[i].getDivDensity());
    //}
    /*
    StringBuffer result = new StringBuffer(" orderedBinList. Counts = ");
    double sum = 0.0;
    if (m_bins != null) {
      for (int i = 0; i < orderedBinList.length; i++) {
	MultiBin bin = (MultiBin) orderedBinList[i];
	double num = bin.getWeight();
	sum += num;
	result.append(" " + Utils.doubleToString(num, 2));
      }
      result.append("  (Total = " + Utils.doubleToString(sum, 2) + " / "
	  + m_bins.size() + " bins).\n");
      result.append("\nVolumes ");
      for (int i = 0; i < orderedBinList.length; i++) {
	MultiBin bin = (MultiBin) orderedBinList[i];
	double num = bin.getVolume();
	result.append(" " + Utils.doubleToString(num, 2));
      }
      result.append("\n\nDensities ");
      for (int i = 0; i < orderedBinList.length; i++) {
	MultiBin bin = (MultiBin) orderedBinList[i];
	double num = bin.getDensity();
	result.append(" " + Utils.doubleToString(num, 2));
      }
   } else {
      result.append(" (No Bins).\n");
    }
    DBO.pln(result.toString());
     */
    int numBins = orderedBinList.length;
    int [] belongsToCentre = new int[numBins];
    int [] isANeighborOf = new int[numBins];
    int [] isCentreNum = new int[numBins];
    for (int i = 0; i < numBins; i++) {
      belongsToCentre[i] = -1;
      isANeighborOf[i] = -1;
      isCentreNum[i] = -1;
    }

    //***check first which are the centres ***************************************************** 
    // DIVDENSITY needs to ignore all with no positive instances
    int firstCentre = 0;
    if (m_compareMethod == DIVDENSITY) {
      // skip over those without positive ones
      double numB = orderedBinList[firstCentre].getNumB_Inst();
      while (numB <= 0 && firstCentre < numBins) {
	firstCentre++;
	if (firstCentre < numBins)
	  numB = orderedBinList[firstCentre].getNumB_Inst();
      }
    }

    // the first one (most dens of the ones with some positive) is a centre     
    if (firstCentre < numBins) {
      centres.add(orderedBinList[firstCentre]);
      isCentreNum[firstCentre] = centres.size() - 1;;
      belongsToCentre[firstCentre] = firstCentre;
    }

    // collect all the centres, start to check at the bin one less dense as first centre
    for (int testCentre = firstCentre + 1; testCentre < numBins; testCentre++) {
      boolean hasDenserNeighbor = false;
      boolean firstSearch = true;
      //DBO.pln("Bin i is "+testCentre);
      for (int j = firstCentre; j < testCentre && !hasDenserNeighbor; j++) {
	if (!MultiBinningUtils.binNotNeighborOf(orderedBinList[testCentre], orderedBinList[j], m_tree,
	    m_originalData.numAttributes())) {
	  //DBO.pln("Bin "+testCentre+" is neighbor of denser bin "+j);
	  hasDenserNeighbor = true;
	} //else
	//DBO.pln("Bin "+testCentre+" is NOT neighbor of "+j);
      } // for j all more dense bins

      // found a centre
      if (!hasDenserNeighbor) {
	belongsToCentre[testCentre] = testCentre;
	isANeighborOf[testCentre] = testCentre;

	// DIVDENSITY cannot have centres with no positive bins
	double numB = orderedBinList[testCentre].getNumB_Inst();
	if ((m_compareMethod != DIVDENSITY)
	    || ((m_compareMethod == DIVDENSITY) && (numB > 0))) {
	  centres.add(orderedBinList[testCentre]);
	  // translate into centre index
	  isCentreNum[testCentre] = centres.size() - 1;
	  // put into centre list
	}
      }
    }

    /*** find neighborhood for all centres found */
    int numCentres = centres.size();
    Vector clusterBinList = new Vector();  
    for (int currCentre = 0; currCentre < numCentres; currCentre++) {

      Vector binList = new Vector();
      // initialize
      for (int i = firstCentre + 1; i < numBins; i++) {
	belongsToCentre[i] = -1;
	isANeighborOf[i] = -1;
	//isCentreNum[i] = -1;
      }      

      // find index of centre in sorted bin list
      int currCentreIndex = 0;
      for (; currCentreIndex < numBins 
      	&& currCentre != isCentreNum[currCentreIndex]; currCentreIndex++) {
	
      }
      belongsToCentre[currCentreIndex] = currCentreIndex;
      // test each less denser then current centre bin
      for (int i = currCentreIndex + 1; i < numBins; i++) {
	boolean hasDenserNeighbor = false;

	// look for neighbors between and including current centre and current bin
	for (int j = currCentreIndex; j < i && !hasDenserNeighbor; j++) {
	  if (!MultiBinningUtils.binNotNeighborOf(orderedBinList[i], orderedBinList[j], m_tree,
	      m_originalData.numAttributes())) {
	    hasDenserNeighbor = true;
	    belongsToCentre[i] = belongsToCentre[j];
	    isANeighborOf[i] = j;
	  } 
	} // for j all more dense bins
      } // for all bins less dens than this centre

      // gather info in lists
      // binList.add(orderedBinList[currCentreIndex]);
      for (int i = currCentreIndex; i < numBins; i++) {
	if (belongsToCentre[i] == currCentreIndex)
	  binList.add(orderedBinList[i]);
      }
      // !!! info in isANeighbor is not used at the moment

      clusterBinList.add(binList);

    } // leftEnd for all centres

    m_centreBins = centres;
    m_clusterBinList = new Vector[numCentres];
    for (int i = 0; i < numCentres; i++) {
      m_clusterBinList[i] = (Vector)clusterBinList.elementAt(i);

    }
  }

  /**
   * Get all the num bins that have the highest density. They don't have to be centres.
   * @param the number of bins asked for, -1 if not set
   * @param the smallest percentage of positive instances allowed, -1.0 if not set 
   */
  public void findDensestBins(int maxNumBins, double minPercent) {

    Vector centres = new Vector();

    // order bins per density, densest first
    MultiBin [] orderedBinList = MultiBinningUtils.sortBinsByDensity(m_bins);
    int numBins = orderedBinList.length;

    // numBins not set?
    if (maxNumBins < 0) maxNumBins = numBins;

    // minPercent not send
    if (minPercent < 0.0) minPercent = 0.0;

    for (int i = 0; i < numBins && i < maxNumBins; i++) {
      MultiBin bin = orderedBinList[i];
      double numNeg = bin.getNumInst(); 
      double numPos = bin.getNumB_Inst();
      double percent = MultiBinningUtils.percent(numNeg + numPos, numPos);
      //if (debug)
      //  DBO.pln("percent of bin "+j+" in cluster "+i+" is "+percent+
      //      " = "+numPos+" positiv instances");
      if (percent >= minPercent) {
	centres.add(bin);
      }
    }

    int numCentres = centres.size();
    m_centreBins = centres;
  }

  /**
   * Get all bins that are high density centres. Adds only the bins to one of the centres
   * @return the vector of bins that are centres
   */
  public void findXORCentreBins(boolean useClusterBinList) {

    Vector centres = new Vector();
    MultiBin [] orderedBinList = MultiBinningUtils.sortBinsByDensity(m_bins);
    //for (int i = 0; i < orderedBinList.length; i++) {
    //DBO.pln(""+orderedBinList[i].getDensity());
    //}
    //DBO.pln("DivDensity");
    //for (int i = 0; i < orderedBinList.length; i++) {
    //  DBO.pln(""+orderedBinList[i].getDivDensity());
    //}

    int numBins = orderedBinList.length;
    int [] belongsToCentre = new int[numBins];
    int [] isANeighborOf = new int[numBins];
    int [] isCentreNum = new int[numBins];
    for (int i = 0; i < numBins; i++) {
      belongsToCentre[i] = -1;
      isANeighborOf[i] = -1;
      isCentreNum[i] = -1;
    }

    // most dens bin is centre
    centres.add(orderedBinList[0]);
    isCentreNum[0] = 0;
    belongsToCentre[0] = 0;

    // find all neighbor and neighbor-neighbor bins of one centre after another
    // start with densest centre

    // check all bins after the densest
    for (int i = 1; i < numBins; i++) {
      boolean hasDenserNeighbor = false;

      // over all denser bins check if denser than current
      for (int j = 0; j < i && !hasDenserNeighbor; j++) {
	if (!MultiBinningUtils.binNotNeighborOf(orderedBinList[i], orderedBinList[j], m_tree,
	    m_originalData.numAttributes())) {
	  hasDenserNeighbor = true;
	  belongsToCentre[i] = belongsToCentre[j];
	  isANeighborOf[i] = j;
	}
      }

      // is a centre itself
      if (!hasDenserNeighbor) {
	//DBO.pln("has not denser neighbor "+ i);
	//if (nextCentre < 0) nextCentre = i;
	isCentreNum[i] = centres.size();
	centres.add(orderedBinList[i]);	  
	belongsToCentre[i] = i;
	isANeighborOf[i] = i;
      }
    }

    int numCentres = centres.size();
    m_centreBins = centres;
    m_clusterBinList = new Vector[numCentres];
    for (int i = 0; i < numCentres; i++) {
      m_clusterBinList[i] = new Vector();
    }

    // gather info in lists
    for (int i = 0; i < numBins; i++) {
      //DBO.pln("i "+i);
      //DBO.pln("belongsToCentre "+belongsToCentre[i]+" numcentres "+numCentres);
      //DBO.pln("orderedBinList "+orderedBinList[i]);
      m_clusterBinList[isCentreNum[belongsToCentre[i]]].add(orderedBinList[i]);
    }
  }

  /**
   * @param args
   */
  public static void main(String[] args) {

    try {
//    DBO.pln("argument 0 "+argv[0]);
//    DBO.pln("argument 1"+argv[1]);
//    DBO.pln("");   

      MultiTUBE est = new MultiTUBE();

      MultiEstimator.buildEstimator((MultiEstimator) est, args, false);      
      System.out.println(est.toString());

    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
  } 
}


