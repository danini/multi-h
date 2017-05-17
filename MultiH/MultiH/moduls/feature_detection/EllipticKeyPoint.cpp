#define _USE_MATH_DEFINES
#include <cmath>
#include "EllipticKeyPoint.h"

EllipticKeyPoint::EllipticKeyPoint() : cv::KeyPoint()
{
	this->transformation = cv::Mat_<double>::eye(2,2);
}

EllipticKeyPoint::~EllipticKeyPoint()
{
}

EllipticKeyPoint::EllipticKeyPoint(const EllipticKeyPoint& ekp) : cv::KeyPoint(ekp)
{
	this->transformation = ekp.transformation.clone();
	this->ownAffinity = ekp.ownAffinity.clone();
}

EllipticKeyPoint::EllipticKeyPoint(const cv::KeyPoint& kp, const cv::Mat_<double> Ai) : cv::KeyPoint(kp)
{
	double rad = kp.size;;
    CV_Assert( rad );
	
	double angle = M_PI * (double)kp.angle/180.;
    double alpha = cos(angle) * rad;
    double beta = sin(angle) * rad;
	cv::Mat M(2, 2, CV_64F);
    double* m = M.ptr<double>();

    m[0] = alpha;
    m[1] = -beta;
    m[2] = beta;
    m[3] = alpha;

	this->transformation = M.clone(); // asszem ez visz a fícsör szpészbe, de lehet hogy az inverze?! nem hiszem...
	
	// apply Ai (utolso elotti egyenlet a pdfben)
	this->pt = applyAffineHomography(Ai, this->pt);
	cv::Mat_<double> Ai_block22 = Ai.colRange(0,2);
	this->transformation = Ai_block22 * this->transformation;
}

cv::Point2d EllipticKeyPoint::applyAffineHomography(const cv::Mat_<double>& H, const cv::Point2d& pt)
{
	return cv::Point2d(((H(0, 0)*pt.x + H(0, 1)*pt.y + H(0, 2))), ((H(1, 0)*pt.x + H(1, 1)*pt.y + H(1, 2))));
}

void EllipticKeyPoint::convert( const std::vector<cv::KeyPoint>& src, std::vector<EllipticKeyPoint>& dst )
{
   /* if( !src.empty() )
    {
        dst.resize(src.size());
        for( size_t i = 0; i < src.size(); i++ )
        {
            dst[i] = EllipticKeyPoint( src[i] );
        }
    }*/
}

void EllipticKeyPoint::convert( const std::vector<EllipticKeyPoint>& src, std::vector<cv::KeyPoint>& dst )
{
    /*if( !src.empty() )
    {
		// TODO
        dst.resize(src.size());
        for( size_t i = 0; i < src.size(); i++ )
        {
            cv::Size_<double> axes = src[i].getAxes();
            double rad = sqrt(axes.height*axes.width);
            dst[i] = cv::KeyPoint(src[i].pt, 2*rad );
        }
	}*/
}

// solveQuadratic needed...
/*cv::Size_<double> EllipticKeyPoint::getAxes() const
{
	auto ellipse = getEllipse();
	double a = ellipse[0], b = ellipse[1], c = ellipse[2];
    double ac_b2 = a*c - b*b;
    double x1, x2;
    solveQuadratic(1., -(a+c), ac_b2, x1, x2);

    double width = double(1./sqrt(x1));
    double height = double(1./sqrt(x2));

	return cv::Size_<double>(width, height);
}

cv::Scalar EllipticKeyPoint::getEllipse() const
{
	return cv::Scalar(transformation(0,0), transformation(1,0), transformation(1,1));
}*/