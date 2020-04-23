/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
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
 * TUBEDD.java
 * formerly known as MITUBEDD.java
 * Copyright (C) 2009 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.mi;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.clusterers.Clusterer;
import weka.clusterers.MultiTUBEClusterer;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.MultiInstanceCapabilitiesHandler;
import weka.core.Optimization;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.Debug.DBO;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.estimators.MultiBin;
import weka.estimators.MultiBinningUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.MultiInstanceToPropositional;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

/**
 <!-- globalinfo-start -->
 * Re-implement the Diverse Density algorithm, augmented with TUBE to find starting points.<br/>
 * <br/>
 * Oded Maron (1998). Learning from ambiguity.<br/>
 * <br/>
 * O. Maron, T. Lozano-Perez (1998). A Framework for Multiple Instance Learning. Neural Information Processing Systems. 10.
 * <p/>
 <!-- globalinfo-leftEnd -->
 * 
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;phdthesis{Maron1998,
 *    author = {Oded Maron},
 *    school = {Massachusetts Institute of Technology},
 *    title = {Learning from ambiguity},
 *    year = {1998}
 * }
 * 
 * &#64;article{Maron1998,
 *    author = {O. Maron and T. Lozano-Perez},
 *    journal = {Neural Information Processing Systems},
 *    title = {A Framework for Multiple Instance Learning},
 *    volume = {10},
 *    year = {1998}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-leftEnd -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -D
 *  Turn on debugging output.</pre>
 * 
 * <pre> -N &lt;num&gt;
 *  Whether to 0=normalize/1=standardize/2=neither.
 *  (default 1=standardize)</pre>
 * 
 <!-- options-leftEnd -->
 *
 * @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
 * contains code copied from MIDD.java
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Xin Xu (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.0 $ 
 */
public class TUBEDD
extends Classifier 
implements OptionHandler, MultiInstanceCapabilitiesHandler,
TechnicalInformationHandler {

  /** for serialization */
  private static final long serialVersionUID = -9039042691334245579L;

  /** additional output element (for verbose modes) */
  DBO dbo = new DBO();

  /** print the info about the centre clusters */
  public static int D_INFOCENTRES                    = 1;

  /** follow the selection of the best cluster */
  public static int D_SHOWSTARTS                     = 2;

  /** follow the selection of the best cluster */
  public static int D_NUMCENTRES                     = 4;

  /** analyze bins with their class differences - long output*/
  public static int D_DIFFAB_WIDEPICT_ALLBINS        = 6;
  
  /** full bin output for all bins*/
  public static int D_FULLBINS_ALLBINS               = 7;
  
  /** analyze bins with their class differences - long output*/
  public static int D_DIFFAB_WIDEPICT_INCENTRES      = 16;
  
  /** output of min and max num of instances in the bags*/
  public static int D_MINMAX_IN_BAGS                 = 21;

  /** output of min and max num of instances in the bags*/
  public static int D_INFO_BAGS                      = 22;

  /** for serialization */

  /** The index of the class attribute */
  protected int m_ClassIndex;

  protected double[] m_Par;

  /** The number of the class labels */
  protected int m_NumClasses;

  /** Class labels for each bag */
  protected int[] m_Classes;

  /** MI data */ 
  protected double[][][] m_Data;

  /** All attribute names */
  protected Instances m_Attributes;

  /** The filter used to standardize/normalize all values. */
  protected Filter m_Filter = null;

  /** Whether to normalize/standardize/neither, default:standardize */
  protected int m_filterType = FILTER_STANDARDIZE;

  /** Normalize training data */
  public static final int FILTER_NORMALIZE = 0;
  /** Standardize training data */
  public static final int FILTER_STANDARDIZE = 1;
  /** No normalization/standardization */
  public static final int FILTER_NONE = 2;
  /** The filter to apply to the training data */
  public static final Tag [] TAGS_FILTER = {
    new Tag(FILTER_NORMALIZE, "Normalize training data"),
    new Tag(FILTER_STANDARDIZE, "Standardize training data"),
    new Tag(FILTER_NONE, "No normalization/standardization"),
  };

  /** The multidimensional binning estimator estimators. */
  //protected MultiEstimator m_estimator = new weka.estimators.MultiTUBE();

  /** filename used for output */
  protected String m_fileName = null;

  /** The filter used to get rid of missing values. */
  protected ReplaceMissingValues m_Missing = new ReplaceMissingValues();

  /** the single-instance weight setting method */
  protected int m_WeightMethod = MultiInstanceToPropositional.WEIGHTMETHOD_INVERSE2;

  /** the percentage a bin is still part of positive cluster */
  double m_minPercent = -1.0;

  /** Filter used to convert MI dataset into single-instance dataset */
  protected MultiInstanceToPropositional m_convertToProp = new MultiInstanceToPropositional();

  /** Clusterer used to find centres */
  protected Clusterer m_clusterer = null;

  /** store info of clusterer */
  protected String m_clusterString = "";

  /** maximal number of bins */
  protected int m_maxNumBins = -1;

  /** all bins found */
  private Vector m_bins = null;

  /** the centre bins */
  private Vector m_centreBins = null;

  /** the cluster bin lists */
  private Vector [] m_clusterBinList = null;

  /** the centrepoint instances */
  private Vector m_centrePoints = null;

  /** random seed */
  int m_seed = 1;
  
  /*=== the start methods  ===*/
  /** take the centre point in the centre bins that has the cluster
   * with the most instances  */
  public static final int S_MOSTINSTANCES            = 1;
  /** take the centre point in the centre bins that has the cluster
   * with the most volume  */
  public static final int S_MOSTVOLUME               = 2;

  /** take all centre points of all centre bins */
  public static final int S_ALLCENTRES               = 3;

  /** take all centre points of all bins over a certain threshhold */
  public static final int S_BINSCRITERIA             = 5;

  /** take a real point as centre point of all centre bins */
  public static final int S_ALLCENTRES_REAL          = 6;

  /** the start method */
  protected int m_startMethod = 3; 

  /** the attribute taken as valid */
  boolean [] m_validAttributes = null;

  /** number of attributes (only valid ones) */
  int m_numAtt = -1;

  /** model used for transformation */
  Instances m_model = null;

  /** vector of best instances */
  protected Vector m_bestInstances = null;

  /** density used for debug wide pict */
  protected double m_maxDensity = Double.NaN;
  protected double m_maxABDensity = Double.NaN;

  /** index of the best instance */
  protected int m_bestInstance = -1;

  /** reduce atts to list of cutting attributes */
  protected boolean m_reduceAtts = false; 

  /** the flags if attribute is valid */
  public TUBEDD () {
    dbo.initializeRanges(30);
    String estString = "weka.clusterers.MultiTUBEClusterer -L last -C 3 -Y 2 -X "+
    "-E \"weka.estimators.MultiTUBE -B 10 -N\"";
    try {
      String [] estSpec = Utils.splitOptions(estString);
      String estName = estSpec[0];
      estSpec[0] = "";
      m_clusterer = (MultiTUBEClusterer) Utils.forName(Clusterer.class, estName, estSpec);
    } catch (Exception ex) {
      ex.printStackTrace();
      DBO.pln(ex.getMessage());
      throw new IllegalStateException("Error while setting clusterer");          
    }
  }

  /**
   * Returns a string describing this filter
   *
   * @return a description of the filter suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return 
    "Re-implement the Diverse Density algorithm, changes the testing "
    + "procedure.\n\n"
    + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    TechnicalInformation 	additional;

    result = new TechnicalInformation(Type.PHDTHESIS);
    result.setValue(Field.AUTHOR, "Oded Maron");
    result.setValue(Field.YEAR, "1998");
    result.setValue(Field.TITLE, "Learning from ambiguity");
    result.setValue(Field.SCHOOL, "Massachusetts Institute of Technology");

    additional = result.add(Type.ARTICLE);
    additional.setValue(Field.AUTHOR, "O. Maron and T. Lozano-Perez");
    additional.setValue(Field.YEAR, "1998");
    additional.setValue(Field.TITLE, "A Framework for Multiple Instance Learning");
    additional.setValue(Field.JOURNAL, "Neural Information Processing Systems");
    additional.setValue(Field.VOLUME, "10");

    return result;
  }

  /**
   * Returns an enumeration describing the available options
   *
   * @return an enumeration of all the available options
   */
  public Enumeration listOptions() {
    Vector result = new Vector();

    result.addElement(new Option(
	"\tTurn on debugging output.",
	"D", 0, "-D"));

    result.addElement(new Option(
	"\tWhether to 0=normalize/1=standardize/2=neither.\n"
	+ "\t(default 1=standardize)",
	"N", 1, "-N <num>"));

    result.addElement(new Option(
	"\tSet the minimal percent value of positive instances in a bin."
	+" with which it is still accepted as part of the positive cluster\n"
	+ "\t(default 99.0)",
	"P", 1, "-P <value>"));

    return result.elements();
  }

  /**
   * Parses a given list of options. <p/>
   *     
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -D
   *  Turn on debugging output.</pre>
   * 
   * <pre> -N &lt;num&gt;
   *  Whether to 0=normalize/1=standardize/2=neither.
   *  (default 1=standardize)</pre>
   * 
   <!-- options-leftEnd -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    // output info data 
    String outputRange = Utils.getOption('V', options);
    setVerboseLevels(outputRange);

    String startString = Utils.getOption('S', options);
    if (startString.length() != 0) {
      setStartMethod(Integer.parseInt(startString));
    }

    String weightString = Utils.getOption('A', options);
    if (weightString.length() != 0) {
      setWeightMethod(
	  new SelectedTag(
	      Integer.parseInt(weightString), 
	      MultiInstanceToPropositional.TAGS_WEIGHTMETHOD));
    } else {
      setWeightMethod(
	  new SelectedTag(
	      MultiInstanceToPropositional.WEIGHTMETHOD_INVERSE2, 
	      MultiInstanceToPropositional.TAGS_WEIGHTMETHOD));
    }	

    setDebug(Utils.getFlag('D', options));

    String nString = Utils.getOption('N', options);
    if (nString.length() != 0) {
      setFilterType(new SelectedTag(Integer.parseInt(nString), TAGS_FILTER));
    } else {
      setFilterType(new SelectedTag(FILTER_STANDARDIZE, TAGS_FILTER));
    }     

    String pString = Utils.getOption('P', options);
    if (pString.length() != 0) {
      setMinPercent(Double.parseDouble(pString));
    }

    String bString = Utils.getOption('B', options);
    if (bString.length() != 0) {
      setMaxNumBins(Integer.parseInt(bString));
    }

    setReduceAtts(Utils.getFlag('R', options));

    // set multidmensional binning estimator and it options
    String [] estSpec = null;
    String estName = "weka.clusterers.MultiTUBEClusterer";
    String estString = Utils.getOption('C', options);
    if (estString.length() != 0) {
      estSpec = Utils.splitOptions(estString);
      if (estSpec.length == 0) {
	throw new IllegalArgumentException("Invalid clusterer specification string");
      }
      estName = estSpec[0];
      estSpec[0] = "";

      setClusterer((Clusterer) Utils.forName(Clusterer.class, estName, estSpec));
    }

    // filename for output of transformed data set
    String fileName = Utils.getOption('O', options);
    if (fileName.length() > 0)
      setFileName(fileName);
  }

  /**
   * Gets the current settings of the classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String[] getOptions() {
    Vector        result;

    result = new Vector();

    if (getDebug())
      result.add("-D");

    result.add("-N");
    result.add("" + m_filterType);

    result.add("-S");
    result.add("" + m_startMethod);

    if (m_minPercent >= 0.0) {
      result.add("-P");
      result.add("" + m_minPercent);
    }

    if (m_maxNumBins >= 0) {
      result.add("-B");
      result.add("" + m_maxNumBins);
    }

    // clusterer
    result.add("-C");
    result.add("" + getClustererSpec());

    // verbose settings
    String verboseLevels = getVerboseLevels();
    if (verboseLevels.length() > 0) {
      result.add("-V");
      result.add("" + verboseLevels);
    }

    if (getReduceAtts())
      result.add("-R");

    return (String[]) result.toArray(new String[result.size()]);
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String filterTypeTipText() {
    return "The filter type for transforming the training data.";
  }

  /**
   * Gets how the training data will be transformed. Will be one of
   * FILTER_NORMALIZE, FILTER_STANDARDIZE, FILTER_NONE.
   *
   * @return the filtering mode
   */
  public SelectedTag getFilterType() {
    return new SelectedTag(m_filterType, TAGS_FILTER);
  }

  /**
   * Sets how the training data will be transformed. Should be one of
   * FILTER_NORMALIZE, FILTER_STANDARDIZE, FILTER_NONE.
   *
   * @param newType the new filtering mode
   */
  public void setFilterType(SelectedTag newType) {

    if (newType.getTags() == TAGS_FILTER) {
      m_filterType = newType.getSelectedTag().getID();
    }
  }

  public int getStartMethod() {
    return m_startMethod;
  }

  public void setStartMethod(int st) {
    m_startMethod = st;
  }

  /**
   * Sets the filename for output
   * @param n the new file name
   */
  public void setFileName(String n) {
    m_fileName = n;
  }

  /**
   * Returns the filename for all info output
   * @return filename that would be used for info output
   */
  public String getFileName() {
    return m_fileName;
  }

  public void setReduceAtts(boolean flag) {
    m_reduceAtts = flag;
  }

  public boolean getReduceAtts() {
    return m_reduceAtts; 
  }

  /**
   * Sets the maximum number of bins the range can be split into
   *
   * @param max the maximum number of splits
   */
  public void setMaxNumBins(int max) {
    m_maxNumBins = max;
  }

  /**
   * Returns the maximum number of bins 
   *
   * @return max the maximum number of binss
   */
  public int getMaxNumBins() {
    return m_maxNumBins;
  }

  /**
   * Switches the outputs on that are requested from the option V
   * if list is empty switches on the verbose mode only
   * 
   * @param list list of integers, all are used for an output type
   */
  public void setVerboseLevels(String list) { 
    dbo.setOutputTypes(list);
  }

  /**
   * Gets the current output type selection
   *
   * @return a string containing a comma separated list of ranges
   */
  public String getVerboseLevels() {
    return dbo.getOutputTypes();
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String weightMethodTipText() {
    return "The method used for weighting the instances.";
  }

  /**
   * The new method for weighting the instances.
   *
   * @param method      the new method
   */
  public void setWeightMethod(SelectedTag method){
    if (method.getTags() == MultiInstanceToPropositional.TAGS_WEIGHTMETHOD)
      m_WeightMethod = method.getSelectedTag().getID();
  }

  /**
   * Returns the current weighting method for instances.
   * 
   * @return the current weighting method
   */
  public SelectedTag getWeightMethod(){
    return new SelectedTag(
	m_WeightMethod, MultiInstanceToPropositional.TAGS_WEIGHTMETHOD);
  }

  /**
   * Sets the estimator
   *
   * @param estimator the estimator with all options set.
   *
  public void setEstimator(MultiEstimator estimator) {

    m_estimator = (MultiEstimator)estimator;
  }

  /**
   * Gets the estimator used.
   *
   * @return the estimator
   *
  public MultiEstimator getEstimator() {

    return m_estimator;
  }

  /**
   * Gets the estimator specification string.
   *
   * @return the estimator string.
   *
  protected String getEstimatorSpec() {

    MultiEstimator e = getEstimator();
    if (e == null) return "";
    if (e instanceof OptionHandler) {
      return e.getClass().getName() + " "
      + Utils.joinOptions(((OptionHandler) e).getOptions());
    }
    return e.getClass().getName();
  }*/

  public double getMinPercent() {
    return m_minPercent;
  }

  public void setMinPercent(double minPercent) {
    m_minPercent = minPercent;
  }

  /**
   * Sets the clusterer
   *
   * @param clusterer the clusterer with all options set.
   */
  public void setClusterer(Clusterer clusterer) {

    m_clusterer = (Clusterer)clusterer;
  }

  /**
   * Gets the clusterer used.
   *
   * @return the clusterer
   */
  public Clusterer getClusterer() {

    return m_clusterer;
  }

  /**
   * Gets the clusterer specification string.
   *
   * @return the estimator string.
   */
  protected String getClustererSpec() {

    Clusterer e = getClusterer();
    if (e == null) return "";
    if (e instanceof OptionHandler) {
      return e.getClass().getName() + " "
      + Utils.joinOptions(((OptionHandler) e).getOptions());
    }
    return e.getClass().getName();
  }

  private class OptEng 
  extends Optimization {

	  /**
	   * Returns the revision string.
	   * 
	   * @return		the revision
	   */
	  public String getRevision() {
	    return RevisionUtils.extract("$Revision: 1.00 $");
	  }

	  /** 
     * Evaluate objective function
     * @param x the current values of variables
     * @return the value of the objective function 
     */
    protected double objectiveFunction(double[] x){
      double nll = 0; // -LogLikelihood
      for(int i = 0; i<m_Classes.length; i++){ // ith bag
	int nI = m_Data[i][0].length; // numInstances in ith bag
	double bag = 0.0;  // NLL of pos bag

	for(int j = 0; j<nI; j++){
	  double ins = 0.0;
	  for(int k = 0; k<m_Data[i].length; k++)
	    ins += (m_Data[i][k][j]-x[k*2])*(m_Data[i][k][j]-x[k*2])*
	    x[k*2+1]*x[k*2+1];
	  ins = Math.exp(-ins);
	  ins = 1.0-ins;

	  if(m_Classes[i] == 1)
	    bag += Math.log(ins);
	  else{
	    if(ins<=m_Zero) ins = m_Zero;
	    nll -= Math.log(ins);
	  }   
	}		

	if(m_Classes[i] == 1){
	  bag = 1.0 - Math.exp(bag);
	  if(bag<=m_Zero) bag = m_Zero;
	  nll -= Math.log(bag);
	}
      }		
      return nll;
    }

    /** 
     * Evaluate Jacobian vector
     * @param x the current values of variables
     * @return the gradient vector 
     */
    protected double[] evaluateGradient(double[] x){
      double[] grad = new double[x.length];
      for(int i = 0; i<m_Classes.length; i++){ // ith bag
	int nI = m_Data[i][0].length; // numInstances in ith bag 

	double denom = 0.0;	
	double[] numrt = new double[x.length];

	for(int j = 0; j<nI; j++){
	  double exp = 0.0;
	  for(int k = 0; k<m_Data[i].length; k++)
	    exp += (m_Data[i][k][j]-x[k*2])*(m_Data[i][k][j]-x[k*2])
	    *x[k*2+1]*x[k*2+1];			
	  exp = Math.exp(-exp);
	  exp = 1.0-exp;
	  if(m_Classes[i]==1)
	    denom += Math.log(exp);		   		    

	  if(exp<=m_Zero) exp = m_Zero;
	  // Instance-wise update
	  for(int p = 0; p<m_Data[i].length; p++){  // pth variable
	    numrt[2*p] += (1.0-exp)*2.0*(x[2*p]-m_Data[i][p][j])*x[p*2+1]*x[p*2+1]
	                                                                    /exp;
	    numrt[2*p+1] += 2.0*(1.0-exp)*(x[2*p]-m_Data[i][p][j])*(x[2*p]-m_Data[i][p][j])
	    *x[p*2+1]/exp;
	  }					    
	}		    

	// Bag-wise update 
	denom = 1.0-Math.exp(denom);
	if(denom <= m_Zero) denom = m_Zero;
	for(int q = 0; q<m_Data[i].length; q++){
	  if(m_Classes[i]==1){
	    grad[2*q] += numrt[2*q]*(1.0-denom)/denom;
	    grad[2*q+1] += numrt[2*q+1]*(1.0-denom)/denom;
	  }else{
	    grad[2*q] -= numrt[2*q];
	    grad[2*q+1] -= numrt[2*q+1];
	  }
	}
      } // one bag

      return grad;
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
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.RELATIONAL_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.BINARY_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // other
    result.enable(Capability.ONLY_MULTIINSTANCE);

    return result;
  }

  /**
   * Returns the capabilities of this multi-instance classifier for the
   * relational data.
   *
   * @return            the capabilities of this object
   * @see               Capabilities
   */
  public Capabilities getMultiInstanceCapabilities() {
    Capabilities result = super.getCapabilities();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.disableAllClasses();
    result.enable(Capability.NO_CLASS);

    return result;
  }

  /**
   * Builds the classifier
   *
   * @param train the training data to be used for generating the
   * boosted classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  public void buildClassifier(Instances train) throws Exception {
    // can classifier handle the data?
    getCapabilities().testWithFail(train);

    // remove instances with missing class
    train = new Instances(train);
    train.deleteWithMissingClass();

    m_ClassIndex = train.classIndex();
    m_NumClasses = train.numClasses();

    int numInst = train.numInstances();
    int numAtts = train.attribute(1).relation().numAttributes();
    m_bestInstances = new Vector();
    int [] bagSize = new int [numInst];
    Instances datasets = new Instances(train.attribute(1).relation(), 0);

    // ***************************************************************************
    // using TUBE clusterer find positive centres and initialize valid atts flags
    m_validAttributes = new boolean[numAtts];
    Instances train2 = findPositiveCentres(train);

    //PrintWriter output = new PrintWriter(new FileOutputStream("/home/gs23/train2.arff"));
    //output.println(train2.toString()); 
    //output.close();

    if (m_Debug) {
      DBO.pln("Extracting data...");
    }
    m_Data = new double [numInst][m_numAtt][];          // Data values
    m_Classes = new int [numInst];                    // Class values
    m_Attributes = datasets.stringFreeStructure();	

    int minBag = Integer.MAX_VALUE;
    int maxBag = 0;
    for(int h = 0; h < numInst; h++)  {  //h_th bag
      Instance current = train.instance(h);
      m_Classes[h] = (int)current.classValue();  // Class value starts from 0

      Instances currInsts = current.relationalValue(1);
      int numInstInBag = currInsts.numInstances();
      if (numInstInBag < minBag) minBag = numInstInBag;
      if (numInstInBag > maxBag) maxBag = numInstInBag;
      
      bagSize[h] = numInstInBag;
      if (dbo.dl(D_INFO_BAGS)) {
	      dbo.pln("bags: "+h+" inst-in-bag: "+numInstInBag+" class "+ m_Classes[h]);
	    }
	     
      for (int i = 0; i < numInstInBag; i++){
	Instance inst = currInsts.instance(i);
	datasets.add(inst);
      }
    }
    if (dbo.dl(D_MINMAX_IN_BAGS)) {
      dbo.pln("min inst in bags: "+minBag+" max: "+maxBag);
    }
         //output = new PrintWriter(new FileOutputStream("/home/gs23/datasets.arff"));
    //output.println(datasets.toString()); 
    //output.close();

    /* filter the training data */
    if (m_Debug) {
      DBO.pln("Filtering data...");
    }
    if (m_filterType == FILTER_STANDARDIZE)  
      m_Filter = new Standardize();
    else if (m_filterType == FILTER_NORMALIZE)
      m_Filter = new Normalize();
    else 
      m_Filter = null; 

    if (m_Filter!=null) {
      m_Filter.setInputFormat(datasets);
      datasets = Filter.useFilter(datasets, m_Filter); 	
    }

    m_Missing.setInputFormat(datasets);
    datasets = Filter.useFilter(datasets, m_Missing);

    //output = new PrintWriter(new FileOutputStream("/home/gs23/datasetstransformed.arff"));
    //output.println(datasets.toString()); 
    //output.close();


    // find the start instances, returned instance has no bag attribute
    m_bestInstances = findStartInstances(train2);

    // int [][]inBin = new int[m_centreBins.length][numPos];

    if (m_Debug) {
      DBO.pln("\nNum of starting instances = " + m_bestInstances.size());
    }
     
    int instIndex = 0;
    int start = 0;	
    for(int h = 0; h < numInst; h++)  {	
      int attIndex = 0;
      for (int i = 0; i < datasets.numAttributes(); i++) {
	if (m_validAttributes[i]) {

	  // initialize m_data[][][]
	  m_Data[h][attIndex] = new double[bagSize[h]];
	  instIndex = start;
	  for (int k = 0; k < bagSize[h]; k++){
	    m_Data[h][attIndex][k] = datasets.instance(instIndex).value(attIndex);
	    instIndex ++;
	  }
	  attIndex++;
	}
      }
      start = instIndex;
    }

    if (m_Debug) {
      DBO.pln("\nIteration History..." );
    }

    double[] x = new double[m_numAtt * 2], tmp = new double[x.length];
    double[][] b = new double[2][x.length]; 

    OptEng opt;
    double nll, bestnll = Double.MAX_VALUE;
    for (int t = 0; t < x.length; t++){
      b[0][t] = Double.NaN; 
      b[1][t] = Double.NaN;
    }

    m_bestInstance = -1;
    int numBest = m_bestInstances.size();
    // best instances
    for(int s = 0; s < numBest; s++){
      // take one of best instances
      Instance inst = (Instance)m_bestInstances.elementAt(s);
      dbo.dpln(D_SHOWSTARTS, "use start instance "+inst);
      inst.setDataset(m_model);
      Instance conv = inst;
      //if (m_Filter!=null) {
      //conv = ((Standardize)m_Filter).convert(inst);
      //}
      if (m_Debug) {
	DBO.pln("converted instance "+conv);
      }
      int attIndex = 0;
      for (int q = 0; q < m_numAtt; q++){
	if (m_validAttributes[q]) {

	  /////////////
	  ///Error///x[2 * attIndex] = conv.value(s);
	  x[2 * attIndex] = conv.value(q);
	  x[2 * attIndex + 1] = 1.0;
	  attIndex++;
	}
      }

      opt = new OptEng();	
      //opt.setDebug(m_Debug);
      tmp = opt.findArgmin(x, b);
      while(tmp==null){
	tmp = opt.getVarbValues();
	if (m_Debug)
	  DBO.pln("200 iterations finished, not enough!");
	tmp = opt.findArgmin(tmp, b);
      }
      nll = opt.getMinFunction();

      if (nll < bestnll){
	m_bestInstance = s;
	bestnll = nll;
	m_Par = tmp;
	if (m_Par == null) {

	  int a = 0;
	}
	tmp = new double[x.length]; // Save memory
	if (m_Debug)
	  DBO.pln("!!!!!!!!!!!!!!!!Smaller NLL found: "+nll);
      }
      if (m_Debug)
	DBO.pln("best instance "+s+":  -------------<Converged>--------------");
    }	

    if (m_Debug) {
      double [] attr = new double [m_Par.length / 2];
      double [] attr2 = new double [m_Par.length / 2];
      for (int j = 0, idx = 0; j < m_Par.length / 2; j++, idx++) {
	attr[j] = m_Par[j * 2];
	attr2[j] = m_Par[j * 2 + 1];
      }
      Instance inst = new Instance(1.0, attr);
      inst.setDataset(datasets);
      Instance inst2 = new Instance(1.0, attr2);
      if (m_Debug) {
	System.out.print( "\n--- point: " + inst);
	// weka.filters.unsupervised.attribute.Standardize
	//***Instance iBack = ((Standardize)m_Filter).convertBackInstance(inst);
	//System.out.print ("\n---  back: " + iBack);
	System.out.print( "\n--- scale: " + inst2);
      }
      //iBack = ((Standardize)m_Filter).convertBackInstance(inst2);
      //System.out.print("--- sca: " + iBack);
    }
  }

  private boolean [] findValidAttributes() {
    boolean valid [] = ((MultiTUBEClusterer)m_clusterer).listOfCuttingAtts();
    return valid;
  }

 /**
  * find the centres of the bins with high diverse density
  * @param train
  * @return
  * @throws Exception
  */
  private Instances findPositiveCentres(Instances train) throws Exception {
    Instances data = new Instances(train);

    // convert the training dataset into single-instance dataset
    try {
      data = convertToPropositional(data);
    } catch (Exception ex) {
      ex.printStackTrace();
      DBO.pln(ex.getMessage());
      throw new IllegalStateException("Error while transforming to propositional");          
    }

    // remove the bag number
    data.deleteAttributeAt(0);

    //String fname = new String("/home/gs23/train1.arff");
    //PrintWriter output = new PrintWriter(new FileOutputStream(fname));
    //output.println(data.toString()); 
    //output.close();

    // the model of this data set
    m_model = new Instances(data, 0);

    // standardize or normalize and treat missing values
    if (m_filterType == FILTER_STANDARDIZE)  
      m_Filter = new Standardize();
    else if (m_filterType == FILTER_NORMALIZE)
      m_Filter = new Normalize();
    else 
      m_Filter = null; 

    if (m_Filter!=null) {
      m_Filter.setInputFormat(data);
      data = Filter.useFilter(data, m_Filter); 	
    }

    m_Missing.setInputFormat(data);
    data = Filter.useFilter(data, m_Missing);

    //DBO.pln(data.toString());
    // ***************************************************************************************
    // build the clusterer
    if (m_startMethod == S_BINSCRITERIA) {
      ((MultiTUBEClusterer)m_clusterer).buildClusterer(data, m_maxNumBins, m_minPercent);
    } else {
      ((MultiTUBEClusterer)m_clusterer).buildClusterer(data);
    }

    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if (dbo.dl(D_DIFFAB_WIDEPICT_ALLBINS)) {

      Vector bins = ((MultiTUBEClusterer)m_clusterer).getBins();
      // get max density
      // m_maxDensity = ((MultiTUBEClusterer)m_clusterer).getMaxDensity();
      // m_maxABDensity = ((MultiTUBEClusterer)m_clusterer).getMaxABDensity();
      m_maxDensity = MultiBinningUtils.getMaxDensity(bins);
      m_maxABDensity = MultiBinningUtils.getMaxABDensity(bins);
      dbo.pln(MultiBinningUtils.binsToPictStringRow(bins, true, true, true, true, true,
	  m_maxDensity, m_maxABDensity, false));
    }

    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if (dbo.dl(D_FULLBINS_ALLBINS)) {
      Vector bins = ((MultiTUBEClusterer)m_clusterer).getBins();
      dbo.pln(MultiBinningUtils.fullBinsToString(bins));
    }
 
    // gather centre infos
    m_centreBins = ((MultiTUBEClusterer)m_clusterer).getCentreBins();
    m_clusterBinList = ((MultiTUBEClusterer)m_clusterer).getClusterBinList();

    // get the centre points, these are the starting points
    // get a real instance as centre point
    if (m_startMethod == S_ALLCENTRES_REAL) {
      Random random = new Random(m_seed);
      m_centrePoints =
	((MultiTUBEClusterer)m_clusterer).getCentrePoints(false, m_centreBins, random);
    } else {
    // get the normal centre points  
      m_centrePoints = 
	((MultiTUBEClusterer)m_clusterer).getCentrePoints(false, m_centreBins);
    }
    m_clusterString = m_clusterer.toString();
    if (dbo.dl(D_DIFFAB_WIDEPICT_INCENTRES)) {
      // get max density
      m_maxDensity = ((MultiTUBEClusterer)m_clusterer).getMaxDensity();
      m_maxABDensity = ((MultiTUBEClusterer)m_clusterer).getMaxABDensity();
      dbo.pln(MultiBinningUtils.binsToPictStringRow(m_centreBins, true, true, true, true, true,
	  m_maxDensity, m_maxABDensity, false));
    }

    // get the list of valid attributes from the clusterer 
    // and recount attributes
    m_numAtt = 0;
    if (getReduceAtts()) {
      boolean[] listofCuttingAtts = ((MultiTUBEClusterer)m_clusterer).getListOfCuttingAtts();
      int valNum = m_validAttributes.length;
      for (int i = 0; i < valNum; i++) {
	m_validAttributes[i] = listofCuttingAtts[i];
	if (m_validAttributes[i]) { m_numAtt++; }
      }
    } else {
      int valNum = m_validAttributes.length;
      m_numAtt = valNum;
      for (int i = 0; i < valNum; i++) {
	m_validAttributes[i] = true;
      }
    }

    // save memory
    m_clusterer = null;

    // delete centres with 0 positive instances
    MultiTUBEClusterer.tidyupCentreList(dbo.dl(D_NUMCENTRES), m_centreBins, m_centrePoints);

    // delete all bins in the bclusters that have less than min percent and are not the centres
    if (m_minPercent > 0.0)
      if (m_clusterBinList != null) 
	MultiTUBEClusterer.tidyupBinLists(dbo.dl(D_NUMCENTRES), m_minPercent, 
	    m_centreBins, m_clusterBinList);

    return data;
  }


  private Vector findStartInstances(Instances data) throws Exception {

    Vector centreInstances = new Vector();
    // simplest, take all centre points
    if (m_startMethod == S_ALLCENTRES || m_startMethod == S_BINSCRITERIA 
	|| m_startMethod == S_ALLCENTRES_REAL) {
      centreInstances = m_centrePoints; 

      dbo.dpln(D_INFOCENTRES, "*** added ALL "+m_centrePoints.size()+"centre points ");
      return centreInstances;
    }
    
    if ((m_startMethod == S_MOSTINSTANCES) || (m_startMethod == S_MOSTVOLUME)) {
      int numCentres = m_centreBins.size();
      int [] numInstInCluster = new int[numCentres];
      double [] volumeInCluster = new double[numCentres];
      int maxNumInst = 0;
      double maxVolume = 0.0;
      int maxNumInstBin = -1;
      int maxVolumeBin = -1;

      dbo.dpln(D_INFOCENTRES, "*** " + numCentres + " centres");

      /** reduce the clusterBinLists to the bins with pos only 
       * or just the centre */
      for (int i = 0; i < numCentres; i++){
	numInstInCluster[i] = 0;
	Vector bins = new Vector();
	MultiBin centreBin = (MultiBin)m_centreBins.elementAt(i);
	bins.add(centreBin);
	numInstInCluster[i] += centreBin.getNumB_Inst();
	volumeInCluster[i] += centreBin.getVolume();
	// start from one, first is centre bin
	dbo.dpln(D_INFOCENTRES, "* cluster: "+i+" started with "+m_clusterBinList[i].size()+" bins");
	for (int j = 1; j < m_clusterBinList[i].size(); j++) {

	  MultiBin bin = (MultiBin)m_clusterBinList[i].elementAt(j);
	  double numNeg = bin.getNumInst();
	  if (numNeg <= 0.0) {
	    bins.add(bin);
	    numInstInCluster[i] += bin.getNumB_Inst();
	  }
	}
	if (numInstInCluster[i] > maxNumInst) {
	  maxNumInst = numInstInCluster[i];
	  maxNumInstBin = i;
	}
	if (volumeInCluster[i] > maxVolume) {
	  maxVolume = volumeInCluster[i];
	  maxVolumeBin = i;
	}

	m_clusterBinList[i] = bins;
	dbo.dpln(D_INFOCENTRES, "*** cluster: "+i+" has "+bins.size()+" bins");
	dbo.dpln(D_INFOCENTRES, "    instances: "+numInstInCluster[i]);
	dbo.dpln(D_INFOCENTRES, "       volume: "+volumeInCluster[i]);
      }

      if (m_startMethod == S_MOSTINSTANCES) {
	dbo.dpln(D_INFOCENTRES, "*** most instances");
	dbo.dpln(D_INFOCENTRES, "*** added centre point from centre "+maxNumInstBin);
	// take the centre of the centre bin in the cluster with the most positive instances
	Instance inst = (Instance)m_centrePoints.elementAt(maxNumInstBin);
	centreInstances.add(inst); 
      }
      if (m_startMethod == S_MOSTVOLUME) {
	dbo.dpln(D_INFOCENTRES, "*** most volume");
	dbo.dpln(D_INFOCENTRES, "*** added centre point from centre "+maxVolumeBin);
	// take the centre of the centre bin in the cluster with the most positive instances
	Instance inst = (Instance)m_centrePoints.elementAt(maxVolumeBin);
	centreInstances.add(inst); 
      }
    }
    return centreInstances;
  }

  /**
   * Converts dataset into single-instance dataset
   * this means one instance per bag
   * @param data the instances to convert
   * @return the converted instances
   * @exception if exception in the filter
   */
  private Instances convertToPropositional(Instances data) throws Exception {
    Instances newInst = new Instances(data, 0);
    m_convertToProp.setWeightMethod(getWeightMethod());
    m_convertToProp.setInputFormat(data);

    newInst = Filter.useFilter(data, m_convertToProp);
    //m_propositionalFormat = new Instances(newInst, 0);
    return newInst;
  }

  /**
   * Computes the distribution for a given exemplar
   *
   * @param exmp the exemplar for which distribution is computed
   * @return the distribution
   * @throws Exception if the distribution can't be computed successfully
   */
  public double[] distributionForInstance(Instance exmp) 
  throws Exception {

    // Extract the data
    Instances ins = exmp.relationalValue(1);
    if(m_Filter!=null)
      ins = Filter.useFilter(ins, m_Filter);

    ins = Filter.useFilter(ins, m_Missing);

    int numInst = ins.numInstances(), fullNumAttr = ins.numAttributes();
    double[][] dat = new double [numInst][m_numAtt];
    for(int j = 0; j < numInst; j++){
      int attrIndex = 0;
      for(int k = 0; k < fullNumAttr; k++){ 
	if (m_validAttributes[k]) {
	  dat[j][attrIndex] = ins.instance(j).value(attrIndex);
	  attrIndex++;
	}
      }
    }

    // Compute the probability of the bag
    double [] distribution = new double[2];
    distribution[0] = 0.0;  // log-Prob. for class 0

    for(int i = 0; i < numInst; i++){
      double exp = 0.0;
      for(int r = 0; r < m_numAtt; r++)
	exp += (m_Par[r*2]-dat[i][r])*(m_Par[r*2]-dat[i][r])*
	m_Par[r*2+1]*m_Par[r*2+1];
      exp = Math.exp(-exp);

      // Prob. updated for one instance
      distribution[0] += Math.log(1.0-exp);
    }

    distribution[0] = Math.exp(distribution[0]);
    distribution[1] = 1.0-distribution[0];

    return distribution;
  }

  /**
   * Gets a string describing the classifier.
   *
   * @return a string describing the classifer built.
   */
  public String toString() {

    //double CSq = m_LLn - m_LL;
    //int df = m_NumPredictors;
    String result = "Multi Instance TUBE Diverse Density";

    result += "\n==\n\nClusterer used: "+ m_clusterString + "\n";
    result += "\nBest start instance was from bin: " + m_bestInstance + "\n\n";
    if (dbo.dl(D_DIFFAB_WIDEPICT_INCENTRES)) {
      result += "The centre bin\n";
      // get max density
      Vector bin = new Vector();
      bin.add((MultiBin)(m_centreBins.elementAt(m_bestInstance)));
      result += MultiBinningUtils.binsToPictStringRow(bin, true, true, true, true, true,
	  m_maxDensity, m_maxABDensity, false);
    }

    if (m_Par == null) {
      return result + ": No model built yet.";
    }

    result += "\nCoefficients...\n" + "Variable       Point       Scale\n";
    for (int j = 0, idx=0; j < m_Par.length/2; j++, idx++) {
      result += m_Attributes.attribute(idx).name();
      result += " "+Utils.doubleToString(m_Par[j*2], 12, 4); 
      result += " "+Utils.doubleToString(m_Par[j*2+1], 12, 4)+"\n";
    }

    return result;
  }

  /**
   * Main method for testing this class.
   *
   * @param argv should contain the command line arguments to the
   * scheme (see Evaluation)
   */
  public static void main(String[] argv) {
    runClassifier(new TUBEDD(), argv);
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 1.00 $");
  }

}
