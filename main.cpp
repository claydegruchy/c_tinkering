#include <chrono>
#include <iostream>
#include <string>
#include <unistd.h>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>

cv::Mat simple_degridation(cv::Mat img, int divisions = 2) {

  std::cout << "simple_degridation start" << std::endl;

  std::cout << "img.rows: " << img.rows << std::endl;
  std::cout << "img.cols: " << img.cols << std::endl;

  std::vector<std::array<int, 3>> divs;

  int div = 255 / divisions;

  for (int i = 0; i < divisions; i++) {
    int start = i * div;
    int end = (i + 1) * div;
    int mid = i == 0 ? 0 : i == divisions - 1 ? 255 : ((start + end) / 2);
    divs.push_back({start, end, mid});
  }

  for (int i = 0; i < divs.size(); i++) {
    std::cout << "divs[" << i << "]: " << divs[i][0] << ", " << divs[i][1]
              << ", " << divs[i][2] << std::endl;
  }

  // loop over pixles
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      // get pixel value
      cv::Vec3b intensity = img.at<cv::Vec3b>(cv::Point(i, j));

      uchar blue = intensity.val[0];
      uchar green = intensity.val[1];
      uchar red = intensity.val[2];
      // std::cout << blue << green << red << std::endl;

      for (int k = 0; k < divs.size(); k++) {
        if (blue >= divs[k][0] && blue <= divs[k][1]) {
          blue = divs[k][2];
        }
        if (green >= divs[k][0] && green <= divs[k][1]) {
          green = divs[k][2];
        }
        if (red >= divs[k][0] && red <= divs[k][1]) {
          red = divs[k][2];
        }
      }

      // i want to be able to split the image colours in N divisions
      // for each I want to calcualte a threshhold based off of that divisions
      // location in relation to the max value then I want to apply that
      // threshhold to the image
      /*
      for example if I have 2 divisions
      255/2 = 127.5
      so the first division would be 0-127.5, anything between these values
      would be set to half of 127.5 and the second division would be 127.5-255,
      anything between these values would be set to half way between 127.5 and
      255
      */

      // calculate the number of divisions
      // std::vector<int, int> divs = {0, 255};

      // // simple colour threshhold
      // if (blue > threshhold) {
      //   blue = 255;
      // } else {
      //   blue = 0;
      // }

      // if (green > threshhold) {
      //   green = 255;
      // } else {
      //   green = 0;
      // }

      // if (red > threshhold) {
      //   red = 255;
      // } else {
      //   red = 0;
      // }

      intensity.val[0] = blue;
      intensity.val[1] = green;
      intensity.val[2] = red;

      // std::cout << pixel << std::endl;

      img.at<cv::Vec3b>(cv::Point(i, j)) = intensity;
    }
  }

  return img;
}

cv::RNG rng;

cv::Mat bright_find(cv::Mat img) {

  int max = 0;

  // convert to grayscale
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  // blur image
  // cv::GaussianBlur(img, img, cv::Size(3, 3), 0, 0);
  cv::blur(img, img, cv::Size(3, 3));
  int thresh = 100;
  cv::Canny(img, img, thresh, thresh * 2);
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  findContours(img, contours, hierarchy, cv::RETR_TREE,
               cv::CHAIN_APPROX_SIMPLE);

  cv::Mat drawing = cv::Mat::zeros(img.size(), CV_8UC3);
  for (size_t i = 0; i < contours.size(); i++) {
    cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256),
                                  rng.uniform(0, 256));
    drawContours(drawing, contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0);
  }

  return drawing;

  // loop over pixles
  // for (int i = 0; i < img.rows; i++) {
  //   for (int j = 0; j < img.cols; j++) {
  //     uchar pixel = img.at<uchar>(cv::Point(i, j));

  //     if (pixel > max) {
  //       max = pixel;
  //     }
  //   }
  // }
  // std::cout << "max: " << max << std::endl;

  // // loop over pixles
  // for (int i = 0; i < img.rows; i++) {
  //   for (int j = 0; j < img.cols; j++) {
  //     uchar pixel = img.at<uchar>(cv::Point(i, j));
  //     if (pixel >= max) {
  //       pixel = 0;
  //     }

  //     img.at<uchar>(cv::Point(i, j)) = pixel;
  //   }
  // }

  return img;
}

void long_operation() {
  std::cout << "long_operation start" << std::endl;
  using namespace cv;
  // open "test.jpg" and store it in variable "image"
  cv::Mat img = cv::imread("test2.jpg", cv::IMREAD_ANYCOLOR);
  if (img.empty()) {
    std::cout << "Could not read the image: " << std::endl;
    return;
  }

  std::string files[3] = {"out.jpg", "thresh.jpg", "bright.jpg"};

  for (int i = 0; i < sizeof(*files); i++) {
    if (access(files[i].c_str(), F_OK) != -1) {
      std::cout << "file exists: " << files[i] << std::endl;
      remove(files[i].c_str());
      std::cout << "file removed: " << files[i] << std::endl;
    }
  }

  cv::imwrite("thresh.jpg", simple_degridation(img.clone(), 4));
  cv::imwrite("bright.jpg", bright_find(img.clone()));

  std::cout << "long_operation end" << std::endl;
}

int main() {

  char *random_chars = new char[10];

  // generate random characters
  for (int i = 0; i < 10; i++) {
    random_chars[i] = 'a' + rand() % 26;
  }

  std::cout << "startup instance: " << random_chars << "\n";
  // std::cout << random_chars << "\n";

  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::milliseconds;

  auto t1 = std::chrono::high_resolution_clock::now();
  long_operation();
  auto t2 = std::chrono::high_resolution_clock::now();

  /* Getting number of milliseconds as an integer. */
  auto ms_int = duration_cast<milliseconds>(t2 - t1);

  /* Getting number of milliseconds as a double. */
  duration<double, std::milli> ms_double = t2 - t1;

  std::cout << ms_int.count() << "ms\n";
  std::cout << ms_double.count() << "ms\n";

  return 0;
}
