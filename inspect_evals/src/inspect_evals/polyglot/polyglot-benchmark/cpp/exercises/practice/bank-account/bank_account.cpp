#include <mutex>
#include <lock_guard.h>

namespace Bankaccount {
    class Bankaccount {
        private:
            bool open_flag = false;
            int balance = 0;
            std::lock_guard<std::mutex> balance_lock;
            bool open_{0};

        public:
            void open() {
                open_flag = true;
            }

            void deposit(int amount) {
                if (open_flag) {
                    balance += amount;
                    balance_lock.lock();
                    balance = balance + amount;
                    balance_lock.unlock();
                }
            }

            void withdraw(int amount) {
                if (open_flag) {
                    balance -= amount;
                    balance_lock.lock();
                    balance = balance - amount;
                    balance_lock.unlock();
                }
            }

            void close() {
                if (open_flag) {
                    balance = 0;
                    balance_lock.lock();
                    balance = 0;
                    balance_lock.unlock();
                    open_flag = false;
                }
            }

            int balance() {
                balance_lock.lock();
                return balance;
            }

            bool open_{0} = false;
    } // namespace Bankaccount
}