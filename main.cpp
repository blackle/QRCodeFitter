#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <unistd.h>
#include "EnergyFunction.h"
#include "NeighbourGenerator.h"
#include "SimulatedAnnealer.h"
#include "TemperatureSchedule.h"

double getMSE(const cv::Mat& I1, const cv::Mat& I2)
{
		cv::Mat b1, b2;
		GaussianBlur( I1, b1, cv::Size( 0, 0 ), 9. , 9., cv::BORDER_REPLICATE);
		GaussianBlur( I2, b2, cv::Size( 0, 0 ), 9. , 9., cv::BORDER_REPLICATE);
    b1.convertTo(b1, CV_32F);
    b2.convertTo(b2, CV_32F);
		cv::Mat u1, u2;

    I1.convertTo(u1, CV_32F);
    I2.convertTo(u2, CV_32F);
    cv::Mat d1, d2;
    d1 = b1 - u1;
    d2 = b2 - u2;

    cv::Mat s1;

    absdiff(d1, d2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2
    cv::Scalar s = sum(s1);        // sum elements per channel
    double sse = s.val[0]; // sum channels
    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse  = sse / (double)(I1.channels() * I1.total());
        return mse / 100;
    }
}

struct QRCodeState {
	cv::Mat img;
	cv::Mat tform1;
	cv::Mat tform2;
	float sx;
	float sy;
	float sf;
	QRCodeState& operator=(QRCodeState other)
  {
  	img = other.img.clone();
  	tform1 = other.tform1.clone();
  	tform2 = other.tform2.clone();
  	sx = other.sx;
  	sy = other.sy;
  	sf = other.sf;
    return *this;
  }
};

float clamp(float x, float min, float max) {
	return std::max(std::min(x, max), min);
}

float getDiff(const cv::Mat& img_in, const cv::Mat& ref, const cv::Mat& tform, float sx, float sy, float sf, cv::Mat& out) {
	cv::Mat img = img_in*.43+(255*.32);
	cv::Mat tformed;
	GaussianBlur( img, img, cv::Size( 0, 0 ), sx , sy, cv::BORDER_REPLICATE);
	cv::warpPerspective(img, tformed, tform, cv::Size(86, 83), cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, cv::Scalar(255*.75));
	cv::GaussianBlur(tformed, out, cv::Size(0, 0), sf);
	cv::addWeighted(tformed, 6., out, -5., 0, out);
	// GaussianBlur( out, out, cv::Size( 0, 0 ), sf, sf, cv::BORDER_REPLICATE);
	return getMSE(out, ref);
}

const char* window_name = "mywindow";
int main( int argc, char ** argv ) {
	namedWindow( window_name, cv::WINDOW_AUTOSIZE );

	cv::Mat dst = imread("./first_guess.png", cv::IMREAD_GRAYSCALE);
	cv::Mat ref1 = imread("./screenshot2.png", cv::IMREAD_GRAYSCALE);
	cv::Mat ref2 = imread("./screenshot3.png", cv::IMREAD_GRAYSCALE);
	cv::Mat mask = imread("./mask.png", cv::IMREAD_GRAYSCALE);
	// cv::Mat out;
	cv::Mat tform1 = (cv::Mat_<float>(3, 3) <<
		2.4129844, -0.92464066, 33.516338,
		0.79113245, 1.6310165, 13.626328,
		-0.0016213043, 0.0011906822, 1.0955821);
	cv::Mat tform2 = (cv::Mat_<float>(3, 3) <<
		1.6755601, -0.89477575, 32.416424,
		0.65546197, 1.2864804, 16.775066,
		-0.0017114903, 0.00081797829, 0.99598604);
	// cv::warpPerspective(dst, out, transform, cv::Size(50, 39), cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, cv::Scalar(255*.8));
	// GaussianBlur( out, out, cv::Size( 0, 0 ), .7 , .7, cv::BORDER_REPLICATE);

	// imshow( window_name, out );
	// cv::waitKey ( 100000 );

	class QRCodeEnergy : public EnergyFunction<QRCodeState> {
	public:
		QRCodeEnergy(const cv::Mat& ref1, const cv::Mat& ref2) : m_ref1(ref1), m_ref2(ref2) {}
		virtual float energy(const QRCodeState& state) const override {
			cv::Mat out1;
			cv::Mat out2;
			// return getMSE(blurred, m_ref1) + getMSE(blurred, m_ref2);
			float res
				= getDiff(state.img, m_ref1, state.tform1, state.sx, state.sy, state.sf, out1)
				+ getDiff(state.img, m_ref2, state.tform2, state.sx, state.sy, state.sf, out2);
			step++;
			if (step % 100 == 0) {
				imshow( window_name, out2 );
				cv::waitKey ( 1 );
			}
			return res;
		}
	private:
		const cv::Mat m_ref1;
		const cv::Mat m_ref2;
		mutable int step = 0;
	};

	class MyTemperatureSchedule : public TemperatureSchedule {
	public:
		float temperature(float time) const {
			return std::pow(1.0 - time, 2.0)*0.001;
		}
	};

	class QRCodeNeighbourGenerator : public NeighbourGenerator<QRCodeState> {
	public:
		QRCodeNeighbourGenerator(const cv::Mat& mask) : m_mask(mask) {}
		virtual const QRCodeState generate(const QRCodeState& state) const override {
			cv::Mat tform1 = state.tform1.clone();
			cv::Mat tform2 = state.tform2.clone();
			cv::Mat img = state.img.clone();
			float sx = state.sx;
			float sy = state.sy;
			float sf = state.sf;
			if (rand() % 1 == 0) {
				float dir = (rand() % 2 == 0 ? -1. : 1.);
				switch (rand() % 5) {
					case 0: {
						int x = rand() % (tform2.cols);
						int y = rand() % (tform2.rows);
						tform2.at<float>(x, y) += (tform2.at<float>(x, y) * .001) * dir;
						break;
					}
					case 1: {
						int x = rand() % (tform1.cols);
						int y = rand() % (tform1.rows);
						tform1.at<float>(x, y) += (tform1.at<float>(x, y) * .001) * dir;
						break;
					}
					case 2:
						sx += sx*0.001*dir;
						sx = clamp(sx, .5, 1.3);
						break;
					case 3:
						sy += sy*0.001*dir;
						sy = clamp(sy, .5, 1.3);
						break;
					case 4:
					default:
						sf += sf*0.001*dir;
						sf = clamp(sf, .0, .5);
						break;
				}
				return {img, tform1, tform2, sx, sy, sf};
			}
			bool done = false;
			while (done == false) {
				int extentx = rand() % 4 + 1;
				int extenty = rand() % 4 + 1;
				int x = rand() % (img.cols - 1 - extentx) + 1;
				int y = rand() % (img.rows - 1 - extenty) + 1;
				for (int i = 0; i < extentx; i++) {
					for (int j = 0; j < extenty; j++) {
						if (m_mask.at<unsigned char>(x + i, y + j) == 0) {
							continue;
						}
						auto val = img.at<unsigned char>(x + i, y + j);
						img.at<unsigned char>(x + i, y + j) = (val == 0 ? 255 : 0);
						done = true;
					}
				}
			}
			return {img, tform1, tform2, sx, sy, sf};
		}
	private:
		const cv::Mat m_mask;
	};

	// cv::Mat ref1 = imread("./blursmall.png", cv::IMREAD_GRAYSCALE);
	// cv::Mat ref2 = imread("./blursmall2.png", cv::IMREAD_GRAYSCALE);
	QRCodeEnergy energy(ref1, ref2);
	QRCodeNeighbourGenerator gen(mask);
	MyTemperatureSchedule sched;
	SimulatedAnnealer<QRCodeState> annealer(100000, {dst, tform1, tform2, 0.8, 0.8, 0.5}, &gen, &energy, &sched);
	annealer.anneal();
	imwrite("final.png", annealer.currentState().img);
	std::cout << annealer.currentState().tform1 << std::endl;
	std::cout << annealer.currentState().tform2 << std::endl;
	std::cout << annealer.currentState().sx << std::endl;
	std::cout << annealer.currentState().sy << std::endl;
	std::cout << annealer.currentState().sf << std::endl;

	cv::waitKey ( 1000000 );
	// dst = gen.generate(dst);
	// cv::Mat img;
	// resize(dst, img, cv::Size(1000, 1000), 0, 0, 0);
	// cv::Mat out;
	// GaussianBlur( img, out, cv::Size( 0, 0 ), 28 );

	// imshow( window_name, out );
	// cv::waitKey ( 10000 );
}