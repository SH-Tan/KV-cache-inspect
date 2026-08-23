#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace boost::posix_time;

namespace gigasecond {
    static double start_seconds = static_cast<double>(std::chrono::system_clock::now().count());
    static const double target_seconds = start_seconds + 1000000000;

    static const boost::posix_time::time_point start = static_cast<posix_time>(start_seconds);
    static const boost::posix_time::time_point end = start + boost::posix_time::seconds(target_seconds);

    std::cout << "The date and time is: " << end.date() << std::endl;
    std::cout << "The time is: " << end.time() << std::endl;

    return end;
}