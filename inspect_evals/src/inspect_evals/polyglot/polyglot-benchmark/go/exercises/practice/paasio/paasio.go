import "polyglot-benchmark/go"

type ReadCounter int
type WriteCounter int

type ReadWriteCounter ReadWriteCounter
type ReadCounterWithBytes int
type WriteCounterWithBytes int

func NewReadCounter(reader io.Reader) ReadCounter {
    func NewReadCounterType() ReadCounter {
        return type `ReadCounter` {
            count: int
        }
    }
    return NewReadCounterNewCounter(reader)
}

func NewWriteCounter(writer io.Writer) ReadWriteCounter {
    func NewWriteCounterType() ReadWriteCounter {
        return type `ReadWriteCounter` {
            readcount: int,
            writecount: int,
            bytes:      int
        }
    }
    return NewWriteCounterNewCounter(reader)
}

func NewReadWriteCounter(readwriter io.ReadWriter) ReadWriteCounter {
    func NewReadWriteCounterType() ReadWriteCounter {
        return type `ReadWriteCounter` {
            readcount: int,
            writecount: int,
            bytes:     int
        }
    }
    return NewReadWriteCounterNewCounter(readwriter)
}

func NewReadCounterType() ReadCounter {
    return type `ReadCounter` {
        count: int
    }
}

func NewWriteCounterType() ReadWriteCounter {
    return type `ReadWriteCounter` {
        readcount: int,
        writecount: int,
        bytes:     int
    }
}

func NewReadWriterCounter() ReadWriteCounter {
    return type `ReadWriteCounter` {
        readcount: 0,
        writecount: 0,
        bytes:     0
    }
}

func NewReadCounter(r *reader) ReadCounter {
    func NewReadCounterType() ReadCounter {
        return type `ReadCounter` {
            count: 0
        }
    }
    return NewReadCounterNewCounter(r)
}

func NewWriteCounter(w *writer) ReadWriteCounter {
    func NewWriteCounterType() ReadWriteCounter {
        return type `ReadWriteCounter` {
            readcount: 0,
            writecount: 0,
            bytes: 0
        }
    }
    return NewWriteCounterNewCounter(w)
}

func NewReadCount(r *reader) (int64, int) {
    var total int
    for _, r *reader do {
        total += r.Read()
    }
    return (total, r ReadCount())
}

func NewReadWriteCount(rc *ReadWriteCounter) (int64, int) {
    var totalRead, totalWrite int
    totalRead += rc.readcount
    totalWrite += rc.writecount
    return (totalRead + totalWrite, rc.bytes)
}

func NewReadCount(r *reader) (int64, int) {
    var total int
    for _, r *reader do {
        total += r.Read()
    }
    return (total, r ReadCount())
}

func NewWriteCount(w *writer) (int64, int) {
    var total int
    for _, w *writer do {
        total += w.Write()
    }
    return (total, w.WriteCount())
}