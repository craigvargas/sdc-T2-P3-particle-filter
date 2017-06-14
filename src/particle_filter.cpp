/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include <cfloat>

//Debugging
#include <fstream>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  // =================================================================================================
  
  // Varibles to generate gaussian dist for each location parameter
  random_device rd;
  mt19937 gen(rd());
  normal_distribution<double> gaus_x(x, std[0]);
  normal_distribution<double> gaus_y(y, std[1]);
  normal_distribution<double> gaus_theta(theta, std[2]);
  
  // Define the number of particles
  num_particles = 50;
  
  // Initialize the particles and fill in the weights vector
  for(int i=0; i<num_particles; i++){
    Particle particle = Particle();
    particle.id = i;
    particle.x = gaus_x(gen);
    particle.y = gaus_y(gen);
    particle.theta = gaus_theta(gen);
    particle.weight = 1.0;
    
    // Append particle to particle vector
    particles.push_back(particle);
    
    // Add a weight to weights
    weights.push_back(1.0);
  }
  
  // Particle filter is now initialized
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  // =================================================================================================

  
  // Set up randome number generator to add noise later
  random_device rd;
  mt19937 gen(rd());

  // Perform calculculation once and reuse reference
  const double delta_yaw = yaw_rate * delta_t;
  
  //Iterate over all particles and perform prediction update
  for(auto &particle: particles){
    //store in a shorter variable name for readabilty
    const double theta = particle.theta;

    //Update particle based on the motion model
    if(fabs(yaw_rate) < 0.00001){
      double distance = velocity*delta_t;
      particle.x += distance*cos(theta);
      particle.y += distance*sin(theta);
    }else{
      particle.x += (velocity/yaw_rate)*(sin(theta + delta_yaw) - sin(theta));
      particle.y += (velocity/yaw_rate)*(cos(theta) - cos(theta + delta_yaw));
      particle.theta += delta_yaw;
    }
    
    // Define distributions for noise
    normal_distribution<double> gaus_x(0, std_pos[0]);
    normal_distribution<double> gaus_y(0, std_pos[1]);
    normal_distribution<double> gaus_theta(0, std_pos[2]);
    
    //Add noise
    particle.x += gaus_x(gen);
    particle.y += gaus_y(gen);
    particle.theta += gaus_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  // =================================================================================================


  // O(m*n) loop
  // ============
  // Iterate over all observations
  for(auto &obs: observations){
    //Initialize min_dist to the max possible value
    double min_dist = DBL_MAX;
    //Iterate through all possible/predicted landmarks
    for(auto pred: predicted){
      //Calc distance between observation and landmark
      double distance = dist(obs.x, obs.y, pred.x, pred.y);
      // Update min_dist and associate the obervation with the predicted landmark
      if(distance < min_dist){
        min_dist = distance;
        obs.id = pred.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  // =================================================================================================


  // Create a vector to store predicted map landmarks as a LandmarkObs object.
  // LandmarkObs object is requred for the dataAssociation function
  std::vector<LandmarkObs> predicted;
  
  // Iterate over all landmarks in the map and convert them to a LandmarkObs object
  for(int i=0; i<map_landmarks.landmark_list.size(); ++i){
    LandmarkObs pred = LandmarkObs();
    pred.x = double(map_landmarks.landmark_list[i].x_f);
    pred.y = double(map_landmarks.landmark_list[i].y_f);
    pred.id = i;
    predicted.push_back(pred);
  }

  // Create an index used to update the weights vector later
  int particle_index = 0;
  // Create a variable to keep track of sum of all weights
  double total_weight = 0;
  
  // Iterate over all particles to calculate the likelihood of the sensor observations
  // given they were observed by that particle
  for(auto &particle: particles){
    particle.weight = 1.0;
    
    // Create varibles to hold data used by SetAssociation (Mainly for debugging)
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    
    // Transform coordinates of each observation based on the current particle's position
    // Transformation is from the particles coordinate system to the map/global coordinate system
    std::vector<LandmarkObs> transformed_obs;
    for(auto obs: observations){
      double cos_theta = cos(particle.theta);
      double sin_theta = sin(particle.theta);
      LandmarkObs transformed = LandmarkObs();
      double trans_x = particle.x + obs.x*cos_theta - obs.y*sin_theta;
      double trans_y = particle.y + obs.x*sin_theta + obs.y*cos_theta;
      transformed.x = trans_x;
      transformed.y = trans_y;
      transformed_obs.push_back(transformed);
    }
    
    // Preform associations
    // Find a specific map landmark to associate each observation with
    // Nearest neighber algorithm is used
    dataAssociation(predicted, transformed_obs);
    
    // Store standard deviations in a shorter variable name for readability
    double s_x = std_landmark[0];
    double s_y = std_landmark[1];
    
    // Create a varible to hold the interim weight calculations
    long double temp_weight = 1.0;
    
    // Iterate over each observation, with transformed coordinates, and calculate the
    // likelihood that the observation was really observed by the current particle
    for(auto &trans_obs: transformed_obs){
      
      // Put store association and location data for debugging purposes
      associations.push_back(trans_obs.id + 1);
      sense_x.push_back(trans_obs.x);
      sense_y.push_back(trans_obs.y);
      
      // Calculate distance between observation and and real assumed map landmark
      double distance = dist(trans_obs.x, trans_obs.y, predicted[trans_obs.id].x, predicted[trans_obs.id].y);

      // Likelihood calculation (Multivariate Gaussian)
      // ===============================================
      //Multiplier
      double mult = 1/(2*M_PI*s_x*s_y);
      //Exponential term 1
      double exp_term1 = trans_obs.x - double(map_landmarks.landmark_list[trans_obs.id].x_f);
      exp_term1 = pow(exp_term1,2);
      exp_term1 /= (pow(s_x,2));
      //Exponential term 2
      double exp_term2 = trans_obs.y - double(map_landmarks.landmark_list[trans_obs.id].y_f);
      exp_term2 = pow(exp_term2,2);
      exp_term2 /= (pow(s_y,2));
      //Multivariate Gaussian Probability
      long double probability = mult*exp((-1/2.0)*(exp_term1 + exp_term2));
      
      // Update temp_weight
      temp_weight *= probability;
    }
    
    // Update particle weight data, increment particle_index,
    particle.weight = temp_weight;
    weights[particle_index] = temp_weight;
    total_weight += particle.weight;
    particle_index++;
    
    // Set associations for debugging purposes
    particle = SetAssociations(particle, associations, sense_x, sense_y);
  }
  
  // Normalize weights
  for(int i=0; i<particles.size(); ++i){
    particles[i].weight /= total_weight;
    weights[i] /= total_weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  // =================================================================================================
  
  // Set up a random number generator
  random_device rd;
  mt19937 gen(rd());
  // Use discrete_distribution to sample from a distribution where weights are used
  // to determine the probability of each outcome
  std::discrete_distribution<> discrete(weights.begin(), weights.end());
  
  // setup a vector to hold the new sample of particles
  std::vector<Particle> resampled;
  
  // Resampling loop: sample n=num_particles times
  for(int i=0; i<num_particles; ++i){
    resampled.push_back(particles[discrete(gen)]);
  }
  
  // assign the new sample to the particles vector for use in the next filter update
  particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


// Start Debugging
// ================
//  ofstream myfile;
//  myfile.open ("debug_init.txt");
//  myfile << "GPS:\n";
//  myfile << x << "\t" << y << "\t" << theta << "\t" << std[0] << "\t" << std[1] << endl << endl;
//  myfile << "id\tx\ty\ttheta\tweight\n";
//  myfile << "Particles:" << endl << endl;

//  for(auto &particle: particles){
//    myfile << particle.id << "\t" << particle.x << "\t" << particle.y << "\t";
//    myfile << particle.theta << "\t" << particle.weight << endl;
//  }
//  myfile.close();
// ================
// End Debugging
