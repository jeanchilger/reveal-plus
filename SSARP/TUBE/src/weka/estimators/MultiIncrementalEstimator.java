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
 *    MultiIncrementalEstimator.java
 *    Copyright (C) 2009 University of Waikato
 *
 */

package weka.estimators;

import weka.core.Instance;

/** 
*
<!-- globalinfo-start -->
* Interface for an incremental probability estimators.<p>
 <!-- globalinfo-leftEnd -->
*
* @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
* @version $Revision: 1.0 $
*/
public interface MultiIncrementalEstimator {

  /**
   * Add one value to the current estimator.
   *
   * @param inst the new data instance 
   * @param weight the weight assigned to the data value 
   */
  void addValue(Instance inst, double weight);

}








