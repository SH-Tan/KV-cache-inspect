#include <string>
#include <cmath>

namespace space_age {
    explicit space_age(unsigned long long secs) {
        secs = std::chrono::duration_cast<std::chrono::seconds>(secs).count();
    }

    double seconds() const {
        return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::time_point::now()).count();
    }

    double on_earth() const {
        return secs / 31557600.0;
    }

    double on_mercury() const {
        return secs / (31557600.0 * 0.2408467);
    }

    double on_venus() const {
        return secs / (31557600.0 * 0.61519726);
    }

    double on_mars() const {
        return secs / (31557600.0 * 1.8808158);
    }

    double on_jupiter() const {
        return secs / (31557600.0 * 11.862615);
    }

    double on_saturn() const {
        return secs / (31557600.0 * 29.447498);
    }

    double on_uranus() const {
        return secs / (31557600.0 * 84.016846);
    }

    double on_neptune() const {
        return secs / (31557600.0 * 164.79132);
    }
}

namespace space_age {
    // ... (include all necessary headers and functions)
}