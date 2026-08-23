import java.util.concurrent.synchronized;
import java.util.concurrent.atomic.AtomicOperation;

class BankAccount {

    private int balance = 0;

    private final synchronized int openTime;

    private final synchronized withdraw(int amount);
    private final synchronized deposit(int amount);

    public int getBalance() throws BankAccountActionInvalidException {
        return balance;
    }

    public void open() throws BankAccountActionInvalidException {
        if (balance < 0) {
            throw new UnsupportedOperationException("Negative balance");
        }
        balance++;
        return balance;
    }

    public void close() throws BankAccountActionInvalidException {
        if (balance < 0) {
            throw new UnsupportedOperationException("Negative balance");
        }
        balance = 0;
    }

    public int deposit(int amount) throws BankAccountActionInvalidException {
        if (balance < amount) {
            throw new UnsupportedOperationException("Insufficient balance");
        }
        balance += amount;
        return balance;
    }

    public int withdraw(int amount) throws BankAccountActionInvalidException {
        if (balance < amount) {
            throw new UnsupportedOperationException("Insufficient balance");
        }
        balance -= amount;
        return balance;
    }

    // This method is not used as it's part of the standard library and not part of the current problem.
}