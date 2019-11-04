
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <set>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> match_candidates; // candidates based on intersection-of-union like bbox-check

    for (auto & m : kptMatches){
        int kpt_index_frame1 = m.queryIdx; // previous frame = query
        int kpt_index_frame2 = m.trainIdx; // current frame  = train
        cv::KeyPoint& kp1 = kptsPrev[kpt_index_frame1];
        cv::KeyPoint& kp2 = kptsCurr[kpt_index_frame2];

        std::vector<int> prev_kp_bboxes;
        // do a intersection-of-union-like check
        if(boundingBox.roi.contains(kp1.pt) && boundingBox.roi.contains(kp2.pt)){
            match_candidates.push_back(m);
        }
    }

    // 1. average euclidean distances between a keypoint to all to all other keypoints, for previous and current frame separately
    // 2. average the averaged distances over all keypoints
    std::vector<double> euclidean_mean_curr, euclidean_mean_prev;
    for(auto & m : match_candidates){
        std::vector<double> euclidean_mean_curr_inner, euclidean_mean_prev_inner;

        int kpt_index_frame1 = m.queryIdx; // previous frame = query
        int kpt_index_frame2 = m.trainIdx; // current frame  = train
        for(auto & m_other : match_candidates){
            int kpt_other_index_frame1 = m_other.queryIdx; // previous frame = query
            int kpt_other_index_frame2 = m_other.trainIdx; // current frame  = train
            double dist_pair_curr = cv::norm(kptsCurr[kpt_index_frame2].pt - kptsCurr[kpt_other_index_frame2].pt);
            double dist_pair_prev = cv::norm(kptsPrev[kpt_index_frame1].pt - kptsPrev[kpt_other_index_frame1].pt);
            euclidean_mean_curr_inner.push_back(dist_pair_curr);
            euclidean_mean_prev_inner.push_back(dist_pair_prev);
        }
        euclidean_mean_curr.push_back(std::accumulate(euclidean_mean_curr_inner.begin(), euclidean_mean_curr_inner.end(), 0.0) / match_candidates.size());
        euclidean_mean_prev.push_back(std::accumulate(euclidean_mean_prev_inner.begin(), euclidean_mean_prev_inner.end(), 0.0) / match_candidates.size());
    }
    double mean_curr = std::accumulate(euclidean_mean_curr.begin(), euclidean_mean_curr.end(), 0.0)/ match_candidates.size();
    double mean_prev = std::accumulate(euclidean_mean_prev.begin(), euclidean_mean_prev.end(), 0.0)/ match_candidates.size();


    const double distance_ratio_TOL_upper = 1.3; // eulidean distance filter tolerance, ratio deviation from mean
    const double distance_ratio_TOL_lower = 0.7; // eulidean distance filter tolerance, ratio deviation from mean


    // assign inliers to bounding box - allow only for distance-ratio exceeds against the means between previous and current of a maximum of distance_ratio_TOL
    for(auto& m :match_candidates)
    {
        int kpt_index_frame1 = m.queryIdx; // previous frame = query
        int kpt_index_frame2 = m.trainIdx; // current frame  = train
        double ratio = cv::norm(kptsCurr[kpt_index_frame2].pt)/cv::norm(kptsPrev[kpt_index_frame1].pt);
        if(ratio >= distance_ratio_TOL_lower * mean_curr/mean_prev && ratio <= distance_ratio_TOL_upper * mean_curr/mean_prev){
            boundingBox.kptMatches.push_back(m);
            boundingBox.keypoints.push_back(kptsCurr[kpt_index_frame2]);
        }
        // otherwise skip, since outlier
    }

    //std::cout << "Number of boundingBox.kptMatches: " << boundingBox.kptMatches.size() << std::endl;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg, bool print)
{
    if(kptMatches.size() == 0)
    {
        TTC = NAN;
        return;
    } 

    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    {
        // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    //double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    // use median instead of mean since it is less sensitive to outliers (mismatches in keypoints)
    long medIndex = floor(distRatios.size() / 2.0);
    double medianDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence


    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);
    if(print) std::cout << "Computed Camera TTC: " << TTC << endl;
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC, bool print)
{
    // frameRate = frames per second
    double dT = 1./frameRate; // time between two measurements in seconds = 1/frameRate

    // compute the median and filter outliers based on Tukey's method using a cleaning parameter
    std::vector<double> minXPrevSorted;
    for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it){
        minXPrevSorted.push_back(it->x);
    }
    // sort by distances
    std::sort(minXPrevSorted.begin(), minXPrevSorted.end());

    // compute the median for previous frame
    double medianPrev = 0.0;
    if(minXPrevSorted.size()%2){
        // take average of lower and upper median
        std::vector<double>::iterator medianPrev_iterator=minXPrevSorted.begin();
        std::advance(medianPrev_iterator, (size_t)((minXPrevSorted.size())/2)-1);
        medianPrev += *medianPrev_iterator;
        std::advance(medianPrev_iterator, 1);
        medianPrev += *medianPrev_iterator;
        medianPrev *= 0.5;
    }
    else{
        std::vector<double>::iterator medianPrev_iterator=minXPrevSorted.begin();
        std::advance(medianPrev_iterator, (size_t)((minXPrevSorted.size()-1)/2));
        medianPrev = *medianPrev_iterator;
    }

    std::vector<double> minXCurrSorted;
    for(auto it=lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); ++it){
        minXCurrSorted.push_back(it->x);
    }
    // sort by distances
    std::sort(minXCurrSorted.begin(), minXCurrSorted.end());

    // compute the median for current frame
    double medianCurr = 0.0;
    if(minXCurrSorted.size()%2){
        // take average of lower and upper median
        std::vector<double>::iterator medianCurr_iterator=minXCurrSorted.begin();
        std::advance(medianCurr_iterator, (size_t)((minXCurrSorted.size())/2)-1);
        medianCurr += *medianCurr_iterator;
        std::advance(medianCurr_iterator, 1);
        medianCurr += *medianCurr_iterator;
        medianCurr *= 0.5;
    }
    else{
        std::vector<double>::iterator medianCurr_iterator=minXCurrSorted.begin();
        std::advance(medianCurr_iterator, (size_t)((minXCurrSorted.size()-1)/2));
        medianCurr = *medianCurr_iterator;
    }


    if(print){
    std::cout << "prev: minimum " << *minXPrevSorted.begin() << std::endl;
    std::cout << "prev: median " << medianPrev << std::endl;

    std::cout << "curr: minimum " << *minXCurrSorted.begin() << std::endl;
    std::cout << "curr: median " << medianCurr << std::endl;
    }


    double minXPrev = *minXPrevSorted.begin();
    double minXCurr = *minXCurrSorted.begin();

    // compute sample standard deviation
    double dev_prev = 0.0;
    for(std::vector<double>::iterator it = minXPrevSorted.begin(); it<minXPrevSorted.end(); ++it )
    {
        dev_prev+= ((*it)-medianPrev)*((*it)-medianPrev);
    }
    dev_prev /= (minXPrevSorted.size()-1);
    dev_prev = sqrt(dev_prev);
    if(print) std::cout << "standard deviation prev " << dev_prev << std::endl;


    double dev_curr = 0.0;
    for(std::vector<double>::iterator it = minXCurrSorted.begin(); it<minXCurrSorted.end(); ++it )
    {
        dev_curr+= ((*it)-medianCurr)*((*it)-medianCurr);
    }
    dev_curr /= (minXCurrSorted.size()-1);
    dev_curr = sqrt(dev_curr);
    if(print) std::cout << "standard deviation curr " << dev_curr << std::endl;

    double outlier_median_deviation_prev = 1.0*dev_prev; // 3.0*dev_prev
    double outlier_median_deviation_curr = 1.0*dev_curr; // 3.0*dev_prev

    // exlude outlier minima based on absolute deviation from median of 3 times the standard deviation
    std::vector<double>::iterator it = minXPrevSorted.begin();
    while(it != minXPrevSorted.end())
    {
        minXPrev = *it; // take the next minimum value
        if(*it>medianPrev-outlier_median_deviation_prev)
            break; // within the confidence region
        it++;
    }
    it = minXCurrSorted.begin();
    while(it != minXCurrSorted.end())
    {
        minXCurr = *it; // take the next minimum value
        if(*it>medianCurr-outlier_median_deviation_curr)
            break; // within the confidence region
        it++;
    }

    if(print) std::cout << "taken min prev " << minXPrev << std::endl;
    if(print) std::cout << "taken min curr " << minXCurr << std::endl;
    

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev-minXCurr);

    if(print) std::cout << "Computed LIDAR-TTC: " << TTC << std::endl;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
    // first: loop all keypoint matches
    //  -> for each keypoint match find the bbox
    //      - in previous frame and
    //      - current frame that enclose the respective keypoints
    //  -> a keypoint might be enclosed in several bounding boxes
    //  -> all such combinations of previous and current bounding boxes are potential bounding-box matches
    //  -> a bounding-box match is a real match if it has the most points (sort pairs by number of enclosed keypoints matches, for each previous frame bbox
    // take the bbox from second frame with highest number of shared keypoints) 

    // key = previous frame!, value = current frame
    std::multimap<int, int> bbox_match_candidate; // an entry is given for a keypoint match (key=bboxId in previous frame which contains keypoint, value = bboxId in current frame which contains keypoint)

    for (auto & m : matches){
        // find keypoint of this match in previous frame and all its enclosing bounding boxes
        int kpt_index_frame1 = m.queryIdx; // previous frame = query
        int kpt_index_frame2 = m.trainIdx; // current frame  = train
        cv::KeyPoint& kp1 = prevFrame.keypoints[kpt_index_frame1];
        cv::KeyPoint& kp2 = currFrame.keypoints[kpt_index_frame2];

        std::vector<int> prev_kp_bboxes;
        for(auto& bbox : prevFrame.boundingBoxes){
            if(bbox.roi.contains(kp1.pt)){
                prev_kp_bboxes.push_back(bbox.boxID);
            }
        }

        std::vector<int> curr_kp_bboxes;
        for(auto& bbox : currFrame.boundingBoxes){
            if(bbox.roi.contains(kp2.pt)){
                curr_kp_bboxes.push_back(bbox.boxID);
            }
        }

        for(const auto& boxid1 : prev_kp_bboxes){
            for(const auto& boxid2 : curr_kp_bboxes){
                bbox_match_candidate.insert(std::make_pair(boxid1, boxid2));
            }
        }
    }

    // count all pairs and sort by their maximum number of occurances
    std::map<std::pair<int, int>, int> count_pairs; // for each match pair specify the number by

    //cout << "#8 : bbox_match_candidate " << bbox_match_candidate.size() << endl;
    for(const auto& match_pair : bbox_match_candidate)
    {
        count_pairs[match_pair]++; // increase counter by one, if not existing so far, create initially with 1
    }

    std::multimap<int, std::pair<int,int>> count_multimap; // setup a multimap / sorted by key=# occurances of match-pairs

    for(auto& count_pair : count_pairs){
        count_multimap.insert(std::make_pair(count_pair.second, count_pair.first));
        //cout << "#8 : count_multimap add " << count_pair.second <<  " (" << count_pair.first.first << "," <<   count_pair.first.second << ")" << endl;
    }

    std::multimap<int, std::pair<int,int>>::iterator it;
    for(it=count_multimap.begin(); it!=count_multimap.end(); it++){
        // note: by forward iterations, an already inserted key value will be overwritten by a pair with value of current frame with a better number of keypoint matches 
        bbBestMatches[it->second.first]=it->second.second;

        //cout << "#8 : bbBestMatches #keypoints: " << it->first << " for pair (" << it->second.first << "," << it->second.second << endl;
        //cout << "#8 : number of bounding box matches: " << bbBestMatches.size() << endl;
    }

    //cout << "#8 : bbBestMatches size " << bbBestMatches.size() << endl;
    for(std::map<int,int>::iterator bestMatchIterator = bbBestMatches.begin(); bestMatchIterator!=bbBestMatches.end(); bestMatchIterator++){
        // note: by reverse iterations, an already inserted key value will be overwritten by a pair with value of current frame with a better number of keypoint matches 
        //cout << "#8 : bbox match (previous bboxId,current bboxId) (" << bestMatchIterator->first << "," << bestMatchIterator->second << ")" << endl;
    }

}
