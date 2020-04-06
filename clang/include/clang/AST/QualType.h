//===- QualType.h - C Language Family Type Representation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// C Language Family Type Representation
///
/// This file defines the QualType pointer wrapper type for qualified types in
/// the C language family.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_QUALTYPE_H
#define LLVM_CLANG_AST_QUALTYPE_H

#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/Twine.h"
#include <string>

namespace clang {

class ExtQuals;
class QualType;
class ConceptDecl;
class TagDecl;
class Type;

enum {
  TypeAlignmentInBits = 4,
  TypeAlignment = 1 << TypeAlignmentInBits
};

namespace serialization {
  template <class T> class AbstractTypeReader;
  template <class T> class AbstractTypeWriter;
}

} // namespace clang

namespace llvm {

  template <typename T>
  struct PointerLikeTypeTraits;
  template<>
  struct PointerLikeTypeTraits< ::clang::Type*> {
    static inline void *getAsVoidPointer(::clang::Type *P) { return P; }

    static inline ::clang::Type *getFromVoidPointer(void *P) {
      return static_cast< ::clang::Type*>(P);
    }

    static constexpr int NumLowBitsAvailable = clang::TypeAlignmentInBits;
  };

  template<>
  struct PointerLikeTypeTraits< ::clang::ExtQuals*> {
    static inline void *getAsVoidPointer(::clang::ExtQuals *P) { return P; }

    static inline ::clang::ExtQuals *getFromVoidPointer(void *P) {
      return static_cast< ::clang::ExtQuals*>(P);
    }

    static constexpr int NumLowBitsAvailable = clang::TypeAlignmentInBits;
  };

} // namespace llvm

namespace clang {

class ASTContext;
template <typename> class CanQual;
class CXXRecordDecl;
class DeclContext;
class EnumDecl;
class Expr;
class ExtQualsTypeCommonBase;
class FunctionDecl;
class IdentifierInfo;
class NamedDecl;
class ObjCInterfaceDecl;
class ObjCProtocolDecl;
class ObjCTypeParamDecl;
struct PrintingPolicy;
class RecordDecl;
class Stmt;
class TagDecl;
class TemplateArgument;
class TemplateArgumentListInfo;
class TemplateArgumentLoc;
class TemplateTypeParmDecl;
class TypedefNameDecl;
class UnresolvedUsingTypenameDecl;

using CanQualType = CanQual<Type>;

// Provide forward declarations for all of the *Type classes.
#define TYPE(Class, Base) class Class##Type;
#include "clang/AST/TypeNodes.inc"

/// The collection of all-type qualifiers we support.
/// Clang supports five independent qualifiers:
/// * C99: const, volatile, and restrict
/// * MS: __unaligned
/// * Embedded C (TR18037): address spaces
/// * Objective C: the GC attributes (none, weak, or strong)
class Qualifiers {
public:
  enum TQ { // NOTE: These flags must be kept in sync with DeclSpec::TQ.
    Const    = 0x1,
    Restrict = 0x2,
    Volatile = 0x4,
    CVRMask = Const | Volatile | Restrict
  };

  enum GC {
    GCNone = 0,
    Weak,
    Strong
  };

  enum ObjCLifetime {
    /// There is no lifetime qualification on this type.
    OCL_None,

    /// This object can be modified without requiring retains or
    /// releases.
    OCL_ExplicitNone,

    /// Assigning into this object requires the old value to be
    /// released and the new value to be retained.  The timing of the
    /// release of the old value is inexact: it may be moved to
    /// immediately after the last known point where the value is
    /// live.
    OCL_Strong,

    /// Reading or writing from this object requires a barrier call.
    OCL_Weak,

    /// Assigning into this object requires a lifetime extension.
    OCL_Autoreleasing
  };

  enum {
    /// The maximum supported address space number.
    /// 23 bits should be enough for anyone.
    MaxAddressSpace = 0x7fffffu,

    /// The width of the "fast" qualifier mask.
    FastWidth = 3,

    /// The fast qualifier mask.
    FastMask = (1 << FastWidth) - 1
  };

  /// Returns the common set of qualifiers while removing them from
  /// the given sets.
  static Qualifiers removeCommonQualifiers(Qualifiers &L, Qualifiers &R) {
    // If both are only CVR-qualified, bit operations are sufficient.
    if (!(L.Mask & ~CVRMask) && !(R.Mask & ~CVRMask)) {
      Qualifiers Q;
      Q.Mask = L.Mask & R.Mask;
      L.Mask &= ~Q.Mask;
      R.Mask &= ~Q.Mask;
      return Q;
    }

    Qualifiers Q;
    unsigned CommonCRV = L.getCVRQualifiers() & R.getCVRQualifiers();
    Q.addCVRQualifiers(CommonCRV);
    L.removeCVRQualifiers(CommonCRV);
    R.removeCVRQualifiers(CommonCRV);

    if (L.getObjCGCAttr() == R.getObjCGCAttr()) {
      Q.setObjCGCAttr(L.getObjCGCAttr());
      L.removeObjCGCAttr();
      R.removeObjCGCAttr();
    }

    if (L.getObjCLifetime() == R.getObjCLifetime()) {
      Q.setObjCLifetime(L.getObjCLifetime());
      L.removeObjCLifetime();
      R.removeObjCLifetime();
    }

    if (L.getAddressSpace() == R.getAddressSpace()) {
      Q.setAddressSpace(L.getAddressSpace());
      L.removeAddressSpace();
      R.removeAddressSpace();
    }
    return Q;
  }

  static Qualifiers fromFastMask(unsigned Mask) {
    Qualifiers Qs;
    Qs.addFastQualifiers(Mask);
    return Qs;
  }

  static Qualifiers fromCVRMask(unsigned CVR) {
    Qualifiers Qs;
    Qs.addCVRQualifiers(CVR);
    return Qs;
  }

  static Qualifiers fromCVRUMask(unsigned CVRU) {
    Qualifiers Qs;
    Qs.addCVRUQualifiers(CVRU);
    return Qs;
  }

  // Deserialize qualifiers from an opaque representation.
  static Qualifiers fromOpaqueValue(unsigned opaque) {
    Qualifiers Qs;
    Qs.Mask = opaque;
    return Qs;
  }

  // Serialize these qualifiers into an opaque representation.
  unsigned getAsOpaqueValue() const {
    return Mask;
  }

  bool hasConst() const { return Mask & Const; }
  bool hasOnlyConst() const { return Mask == Const; }
  void removeConst() { Mask &= ~Const; }
  void addConst() { Mask |= Const; }

  bool hasVolatile() const { return Mask & Volatile; }
  bool hasOnlyVolatile() const { return Mask == Volatile; }
  void removeVolatile() { Mask &= ~Volatile; }
  void addVolatile() { Mask |= Volatile; }

  bool hasRestrict() const { return Mask & Restrict; }
  bool hasOnlyRestrict() const { return Mask == Restrict; }
  void removeRestrict() { Mask &= ~Restrict; }
  void addRestrict() { Mask |= Restrict; }

  bool hasCVRQualifiers() const { return getCVRQualifiers(); }
  unsigned getCVRQualifiers() const { return Mask & CVRMask; }
  unsigned getCVRUQualifiers() const { return Mask & (CVRMask | UMask); }

  void setCVRQualifiers(unsigned mask) {
    assert(!(mask & ~CVRMask) && "bitmask contains non-CVR bits");
    Mask = (Mask & ~CVRMask) | mask;
  }
  void removeCVRQualifiers(unsigned mask) {
    assert(!(mask & ~CVRMask) && "bitmask contains non-CVR bits");
    Mask &= ~mask;
  }
  void removeCVRQualifiers() {
    removeCVRQualifiers(CVRMask);
  }
  void addCVRQualifiers(unsigned mask) {
    assert(!(mask & ~CVRMask) && "bitmask contains non-CVR bits");
    Mask |= mask;
  }
  void addCVRUQualifiers(unsigned mask) {
    assert(!(mask & ~CVRMask & ~UMask) && "bitmask contains non-CVRU bits");
    Mask |= mask;
  }

  bool hasUnaligned() const { return Mask & UMask; }
  void setUnaligned(bool flag) {
    Mask = (Mask & ~UMask) | (flag ? UMask : 0);
  }
  void removeUnaligned() { Mask &= ~UMask; }
  void addUnaligned() { Mask |= UMask; }

  bool hasObjCGCAttr() const { return Mask & GCAttrMask; }
  GC getObjCGCAttr() const { return GC((Mask & GCAttrMask) >> GCAttrShift); }
  void setObjCGCAttr(GC type) {
    Mask = (Mask & ~GCAttrMask) | (type << GCAttrShift);
  }
  void removeObjCGCAttr() { setObjCGCAttr(GCNone); }
  void addObjCGCAttr(GC type) {
    assert(type);
    setObjCGCAttr(type);
  }
  Qualifiers withoutObjCGCAttr() const {
    Qualifiers qs = *this;
    qs.removeObjCGCAttr();
    return qs;
  }
  Qualifiers withoutObjCLifetime() const {
    Qualifiers qs = *this;
    qs.removeObjCLifetime();
    return qs;
  }
  Qualifiers withoutAddressSpace() const {
    Qualifiers qs = *this;
    qs.removeAddressSpace();
    return qs;
  }

  bool hasObjCLifetime() const { return Mask & LifetimeMask; }
  ObjCLifetime getObjCLifetime() const {
    return ObjCLifetime((Mask & LifetimeMask) >> LifetimeShift);
  }
  void setObjCLifetime(ObjCLifetime type) {
    Mask = (Mask & ~LifetimeMask) | (type << LifetimeShift);
  }
  void removeObjCLifetime() { setObjCLifetime(OCL_None); }
  void addObjCLifetime(ObjCLifetime type) {
    assert(type);
    assert(!hasObjCLifetime());
    Mask |= (type << LifetimeShift);
  }

  /// True if the lifetime is neither None or ExplicitNone.
  bool hasNonTrivialObjCLifetime() const {
    ObjCLifetime lifetime = getObjCLifetime();
    return (lifetime > OCL_ExplicitNone);
  }

  /// True if the lifetime is either strong or weak.
  bool hasStrongOrWeakObjCLifetime() const {
    ObjCLifetime lifetime = getObjCLifetime();
    return (lifetime == OCL_Strong || lifetime == OCL_Weak);
  }

  bool hasAddressSpace() const { return Mask & AddressSpaceMask; }
  LangAS getAddressSpace() const {
    return static_cast<LangAS>(Mask >> AddressSpaceShift);
  }
  bool hasTargetSpecificAddressSpace() const {
    return isTargetAddressSpace(getAddressSpace());
  }
  /// Get the address space attribute value to be printed by diagnostics.
  unsigned getAddressSpaceAttributePrintValue() const {
    auto Addr = getAddressSpace();
    // This function is not supposed to be used with language specific
    // address spaces. If that happens, the diagnostic message should consider
    // printing the QualType instead of the address space value.
    assert(Addr == LangAS::Default || hasTargetSpecificAddressSpace());
    if (Addr != LangAS::Default)
      return toTargetAddressSpace(Addr);
    // TODO: The diagnostic messages where Addr may be 0 should be fixed
    // since it cannot differentiate the situation where 0 denotes the default
    // address space or user specified __attribute__((address_space(0))).
    return 0;
  }
  void setAddressSpace(LangAS space) {
    assert((unsigned)space <= MaxAddressSpace);
    Mask = (Mask & ~AddressSpaceMask)
         | (((uint32_t) space) << AddressSpaceShift);
  }
  void removeAddressSpace() { setAddressSpace(LangAS::Default); }
  void addAddressSpace(LangAS space) {
    assert(space != LangAS::Default);
    setAddressSpace(space);
  }

  // Fast qualifiers are those that can be allocated directly
  // on a QualType object.
  bool hasFastQualifiers() const { return getFastQualifiers(); }
  unsigned getFastQualifiers() const { return Mask & FastMask; }
  void setFastQualifiers(unsigned mask) {
    assert(!(mask & ~FastMask) && "bitmask contains non-fast qualifier bits");
    Mask = (Mask & ~FastMask) | mask;
  }
  void removeFastQualifiers(unsigned mask) {
    assert(!(mask & ~FastMask) && "bitmask contains non-fast qualifier bits");
    Mask &= ~mask;
  }
  void removeFastQualifiers() {
    removeFastQualifiers(FastMask);
  }
  void addFastQualifiers(unsigned mask) {
    assert(!(mask & ~FastMask) && "bitmask contains non-fast qualifier bits");
    Mask |= mask;
  }

  /// Return true if the set contains any qualifiers which require an ExtQuals
  /// node to be allocated.
  bool hasNonFastQualifiers() const { return Mask & ~FastMask; }
  Qualifiers getNonFastQualifiers() const {
    Qualifiers Quals = *this;
    Quals.setFastQualifiers(0);
    return Quals;
  }

  /// Return true if the set contains any qualifiers.
  bool hasQualifiers() const { return Mask; }
  bool empty() const { return !Mask; }

  /// Add the qualifiers from the given set to this set.
  void addQualifiers(Qualifiers Q) {
    // If the other set doesn't have any non-boolean qualifiers, just
    // bit-or it in.
    if (!(Q.Mask & ~CVRMask))
      Mask |= Q.Mask;
    else {
      Mask |= (Q.Mask & CVRMask);
      if (Q.hasAddressSpace())
        addAddressSpace(Q.getAddressSpace());
      if (Q.hasObjCGCAttr())
        addObjCGCAttr(Q.getObjCGCAttr());
      if (Q.hasObjCLifetime())
        addObjCLifetime(Q.getObjCLifetime());
    }
  }

  /// Remove the qualifiers from the given set from this set.
  void removeQualifiers(Qualifiers Q) {
    // If the other set doesn't have any non-boolean qualifiers, just
    // bit-and the inverse in.
    if (!(Q.Mask & ~CVRMask))
      Mask &= ~Q.Mask;
    else {
      Mask &= ~(Q.Mask & CVRMask);
      if (getObjCGCAttr() == Q.getObjCGCAttr())
        removeObjCGCAttr();
      if (getObjCLifetime() == Q.getObjCLifetime())
        removeObjCLifetime();
      if (getAddressSpace() == Q.getAddressSpace())
        removeAddressSpace();
    }
  }

  /// Add the qualifiers from the given set to this set, given that
  /// they don't conflict.
  void addConsistentQualifiers(Qualifiers qs) {
    assert(getAddressSpace() == qs.getAddressSpace() ||
           !hasAddressSpace() || !qs.hasAddressSpace());
    assert(getObjCGCAttr() == qs.getObjCGCAttr() ||
           !hasObjCGCAttr() || !qs.hasObjCGCAttr());
    assert(getObjCLifetime() == qs.getObjCLifetime() ||
           !hasObjCLifetime() || !qs.hasObjCLifetime());
    Mask |= qs.Mask;
  }

  /// Returns true if address space A is equal to or a superset of B.
  /// OpenCL v2.0 defines conversion rules (OpenCLC v2.0 s6.5.5) and notion of
  /// overlapping address spaces.
  /// CL1.1 or CL1.2:
  ///   every address space is a superset of itself.
  /// CL2.0 adds:
  ///   __generic is a superset of any address space except for __constant.
  static bool isAddressSpaceSupersetOf(LangAS A, LangAS B) {
    // Address spaces must match exactly.
    return A == B ||
           // Otherwise in OpenCLC v2.0 s6.5.5: every address space except
           // for __constant can be used as __generic.
           (A == LangAS::opencl_generic && B != LangAS::opencl_constant) ||
           // Consider pointer size address spaces to be equivalent to default.
           ((isPtrSizeAddressSpace(A) || A == LangAS::Default) &&
            (isPtrSizeAddressSpace(B) || B == LangAS::Default));
  }

  /// Returns true if the address space in these qualifiers is equal to or
  /// a superset of the address space in the argument qualifiers.
  bool isAddressSpaceSupersetOf(Qualifiers other) const {
    return isAddressSpaceSupersetOf(getAddressSpace(), other.getAddressSpace());
  }

  /// Determines if these qualifiers compatibly include another set.
  /// Generally this answers the question of whether an object with the other
  /// qualifiers can be safely used as an object with these qualifiers.
  bool compatiblyIncludes(Qualifiers other) const {
    return isAddressSpaceSupersetOf(other) &&
           // ObjC GC qualifiers can match, be added, or be removed, but can't
           // be changed.
           (getObjCGCAttr() == other.getObjCGCAttr() || !hasObjCGCAttr() ||
            !other.hasObjCGCAttr()) &&
           // ObjC lifetime qualifiers must match exactly.
           getObjCLifetime() == other.getObjCLifetime() &&
           // CVR qualifiers may subset.
           (((Mask & CVRMask) | (other.Mask & CVRMask)) == (Mask & CVRMask)) &&
           // U qualifier may superset.
           (!other.hasUnaligned() || hasUnaligned());
  }

  /// Determines if these qualifiers compatibly include another set of
  /// qualifiers from the narrow perspective of Objective-C ARC lifetime.
  ///
  /// One set of Objective-C lifetime qualifiers compatibly includes the other
  /// if the lifetime qualifiers match, or if both are non-__weak and the
  /// including set also contains the 'const' qualifier, or both are non-__weak
  /// and one is None (which can only happen in non-ARC modes).
  bool compatiblyIncludesObjCLifetime(Qualifiers other) const {
    if (getObjCLifetime() == other.getObjCLifetime())
      return true;

    if (getObjCLifetime() == OCL_Weak || other.getObjCLifetime() == OCL_Weak)
      return false;

    if (getObjCLifetime() == OCL_None || other.getObjCLifetime() == OCL_None)
      return true;

    return hasConst();
  }

  /// Determine whether this set of qualifiers is a strict superset of
  /// another set of qualifiers, not considering qualifier compatibility.
  bool isStrictSupersetOf(Qualifiers Other) const;

  bool operator==(Qualifiers Other) const { return Mask == Other.Mask; }
  bool operator!=(Qualifiers Other) const { return Mask != Other.Mask; }

  explicit operator bool() const { return hasQualifiers(); }

  Qualifiers &operator+=(Qualifiers R) {
    addQualifiers(R);
    return *this;
  }

  // Union two qualifier sets.  If an enumerated qualifier appears
  // in both sets, use the one from the right.
  friend Qualifiers operator+(Qualifiers L, Qualifiers R) {
    L += R;
    return L;
  }

  Qualifiers &operator-=(Qualifiers R) {
    removeQualifiers(R);
    return *this;
  }

  /// Compute the difference between two qualifier sets.
  friend Qualifiers operator-(Qualifiers L, Qualifiers R) {
    L -= R;
    return L;
  }

  std::string getAsString() const;
  std::string getAsString(const PrintingPolicy &Policy) const;

  static std::string getAddrSpaceAsString(LangAS AS);

  bool isEmptyWhenPrinted(const PrintingPolicy &Policy) const;
  void print(raw_ostream &OS, const PrintingPolicy &Policy,
             bool appendSpaceIfNonEmpty = false) const;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(Mask);
  }

private:
  // bits:     |0 1 2|3|4 .. 5|6  ..  8|9   ...   31|
  //           |C R V|U|GCAttr|Lifetime|AddressSpace|
  uint32_t Mask = 0;

  static const uint32_t UMask = 0x8;
  static const uint32_t UShift = 3;
  static const uint32_t GCAttrMask = 0x30;
  static const uint32_t GCAttrShift = 4;
  static const uint32_t LifetimeMask = 0x1C0;
  static const uint32_t LifetimeShift = 6;
  static const uint32_t AddressSpaceMask =
      ~(CVRMask | UMask | GCAttrMask | LifetimeMask);
  static const uint32_t AddressSpaceShift = 9;
};

/// A std::pair-like structure for storing a qualified type split
/// into its local qualifiers and its locally-unqualified type.
struct SplitQualType {
  /// The locally-unqualified type.
  const Type *Ty = nullptr;

  /// The local qualifiers.
  Qualifiers Quals;

  SplitQualType() = default;
  SplitQualType(const Type *ty, Qualifiers qs) : Ty(ty), Quals(qs) {}

  SplitQualType getSingleStepDesugaredType() const; // end of this file

  // Make std::tie work.
  std::pair<const Type *,Qualifiers> asPair() const {
    return std::pair<const Type *, Qualifiers>(Ty, Quals);
  }

  friend bool operator==(SplitQualType a, SplitQualType b) {
    return a.Ty == b.Ty && a.Quals == b.Quals;
  }
  friend bool operator!=(SplitQualType a, SplitQualType b) {
    return a.Ty != b.Ty || a.Quals != b.Quals;
  }
};

/// The kind of type we are substituting Objective-C type arguments into.
///
/// The kind of substitution affects the replacement of type parameters when
/// no concrete type information is provided, e.g., when dealing with an
/// unspecialized type.
enum class ObjCSubstitutionContext {
  /// An ordinary type.
  Ordinary,

  /// The result type of a method or function.
  Result,

  /// The parameter type of a method or function.
  Parameter,

  /// The type of a property.
  Property,

  /// The superclass of a type.
  Superclass,
};

/// A (possibly-)qualified type.
///
/// For efficiency, we don't store CV-qualified types as nodes on their
/// own: instead each reference to a type stores the qualifiers.  This
/// greatly reduces the number of nodes we need to allocate for types (for
/// example we only need one for 'int', 'const int', 'volatile int',
/// 'const volatile int', etc).
///
/// As an added efficiency bonus, instead of making this a pair, we
/// just store the two bits we care about in the low bits of the
/// pointer.  To handle the packing/unpacking, we make QualType be a
/// simple wrapper class that acts like a smart pointer.  A third bit
/// indicates whether there are extended qualifiers present, in which
/// case the pointer points to a special structure.
class QualType {
  friend class QualifierCollector;

  // Thankfully, these are efficiently composable.
  llvm::PointerIntPair<llvm::PointerUnion<const Type *, const ExtQuals *>,
                       Qualifiers::FastWidth> Value;

  const ExtQuals *getExtQualsUnsafe() const {
    return Value.getPointer().get<const ExtQuals*>();
  }

  const Type *getTypePtrUnsafe() const {
    return Value.getPointer().get<const Type*>();
  }

  const ExtQualsTypeCommonBase *getCommonPtr() const {
    assert(!isNull() && "Cannot retrieve a NULL type pointer");
    auto CommonPtrVal = reinterpret_cast<uintptr_t>(Value.getOpaqueValue());
    CommonPtrVal &= ~(uintptr_t)((1 << TypeAlignmentInBits) - 1);
    return reinterpret_cast<ExtQualsTypeCommonBase*>(CommonPtrVal);
  }

public:
  QualType() = default;
  QualType(const Type *Ptr, unsigned Quals) : Value(Ptr, Quals) {}
  QualType(const ExtQuals *Ptr, unsigned Quals) : Value(Ptr, Quals) {}

  unsigned getLocalFastQualifiers() const { return Value.getInt(); }
  void setLocalFastQualifiers(unsigned Quals) { Value.setInt(Quals); }

  /// Retrieves a pointer to the underlying (unqualified) type.
  ///
  /// This function requires that the type not be NULL. If the type might be
  /// NULL, use the (slightly less efficient) \c getTypePtrOrNull().
  const Type *getTypePtr() const;

  const Type *getTypePtrOrNull() const;

  /// Retrieves a pointer to the name of the base type.
  const IdentifierInfo *getBaseTypeIdentifier() const;

  /// Divides a QualType into its unqualified type and a set of local
  /// qualifiers.
  SplitQualType split() const;

  void *getAsOpaquePtr() const { return Value.getOpaqueValue(); }

  static QualType getFromOpaquePtr(const void *Ptr) {
    QualType T;
    T.Value.setFromOpaqueValue(const_cast<void*>(Ptr));
    return T;
  }

  const Type &operator*() const {
    return *getTypePtr();
  }

  const Type *operator->() const {
    return getTypePtr();
  }

  bool isCanonical() const;
  bool isCanonicalAsParam() const;

  /// Return true if this QualType doesn't point to a type yet.
  bool isNull() const {
    return Value.getPointer().isNull();
  }

  /// Determine whether this particular QualType instance has the
  /// "const" qualifier set, without looking through typedefs that may have
  /// added "const" at a different level.
  bool isLocalConstQualified() const {
    return (getLocalFastQualifiers() & Qualifiers::Const);
  }

  /// Determine whether this type is const-qualified.
  bool isConstQualified() const;

  /// Determine whether this particular QualType instance has the
  /// "restrict" qualifier set, without looking through typedefs that may have
  /// added "restrict" at a different level.
  bool isLocalRestrictQualified() const {
    return (getLocalFastQualifiers() & Qualifiers::Restrict);
  }

  /// Determine whether this type is restrict-qualified.
  bool isRestrictQualified() const;

  /// Determine whether this particular QualType instance has the
  /// "volatile" qualifier set, without looking through typedefs that may have
  /// added "volatile" at a different level.
  bool isLocalVolatileQualified() const {
    return (getLocalFastQualifiers() & Qualifiers::Volatile);
  }

  /// Determine whether this type is volatile-qualified.
  bool isVolatileQualified() const;

  /// Determine whether this particular QualType instance has any
  /// qualifiers, without looking through any typedefs that might add
  /// qualifiers at a different level.
  bool hasLocalQualifiers() const {
    return getLocalFastQualifiers() || hasLocalNonFastQualifiers();
  }

  /// Determine whether this type has any qualifiers.
  bool hasQualifiers() const;

  /// Determine whether this particular QualType instance has any
  /// "non-fast" qualifiers, e.g., those that are stored in an ExtQualType
  /// instance.
  bool hasLocalNonFastQualifiers() const {
    return Value.getPointer().is<const ExtQuals*>();
  }

  /// Retrieve the set of qualifiers local to this particular QualType
  /// instance, not including any qualifiers acquired through typedefs or
  /// other sugar.
  Qualifiers getLocalQualifiers() const;

  /// Retrieve the set of qualifiers applied to this type.
  Qualifiers getQualifiers() const;

  /// Retrieve the set of CVR (const-volatile-restrict) qualifiers
  /// local to this particular QualType instance, not including any qualifiers
  /// acquired through typedefs or other sugar.
  unsigned getLocalCVRQualifiers() const {
    return getLocalFastQualifiers();
  }

  /// Retrieve the set of CVR (const-volatile-restrict) qualifiers
  /// applied to this type.
  unsigned getCVRQualifiers() const;

  bool isConstant(const ASTContext& Ctx) const {
    return QualType::isConstant(*this, Ctx);
  }

  /// Determine whether this is a Plain Old Data (POD) type (C++ 3.9p10).
  bool isPODType(const ASTContext &Context) const;

  /// Return true if this is a POD type according to the rules of the C++98
  /// standard, regardless of the current compilation's language.
  bool isCXX98PODType(const ASTContext &Context) const;

  /// Return true if this is a POD type according to the more relaxed rules
  /// of the C++11 standard, regardless of the current compilation's language.
  /// (C++0x [basic.types]p9). Note that, unlike
  /// CXXRecordDecl::isCXX11StandardLayout, this takes DRs into account.
  bool isCXX11PODType(const ASTContext &Context) const;

  /// Return true if this is a trivial type per (C++0x [basic.types]p9)
  bool isTrivialType(const ASTContext &Context) const;

  /// Return true if this is a trivially copyable type (C++0x [basic.types]p9)
  bool isTriviallyCopyableType(const ASTContext &Context) const;


  /// Returns true if it is a class and it might be dynamic.
  bool mayBeDynamicClass() const;

  /// Returns true if it is not a class or if the class might not be dynamic.
  bool mayBeNotDynamicClass() const;

  // Don't promise in the API that anything besides 'const' can be
  // easily added.

  /// Add the `const` type qualifier to this QualType.
  void addConst() {
    addFastQualifiers(Qualifiers::Const);
  }
  QualType withConst() const {
    return withFastQualifiers(Qualifiers::Const);
  }

  /// Add the `volatile` type qualifier to this QualType.
  void addVolatile() {
    addFastQualifiers(Qualifiers::Volatile);
  }
  QualType withVolatile() const {
    return withFastQualifiers(Qualifiers::Volatile);
  }

  /// Add the `restrict` qualifier to this QualType.
  void addRestrict() {
    addFastQualifiers(Qualifiers::Restrict);
  }
  QualType withRestrict() const {
    return withFastQualifiers(Qualifiers::Restrict);
  }

  QualType withCVRQualifiers(unsigned CVR) const {
    return withFastQualifiers(CVR);
  }

  void addFastQualifiers(unsigned TQs) {
    assert(!(TQs & ~Qualifiers::FastMask)
           && "non-fast qualifier bits set in mask!");
    Value.setInt(Value.getInt() | TQs);
  }

  void removeLocalConst();
  void removeLocalVolatile();
  void removeLocalRestrict();
  void removeLocalCVRQualifiers(unsigned Mask);

  void removeLocalFastQualifiers() { Value.setInt(0); }
  void removeLocalFastQualifiers(unsigned Mask) {
    assert(!(Mask & ~Qualifiers::FastMask) && "mask has non-fast qualifiers");
    Value.setInt(Value.getInt() & ~Mask);
  }

  // Creates a type with the given qualifiers in addition to any
  // qualifiers already on this type.
  QualType withFastQualifiers(unsigned TQs) const {
    QualType T = *this;
    T.addFastQualifiers(TQs);
    return T;
  }

  // Creates a type with exactly the given fast qualifiers, removing
  // any existing fast qualifiers.
  QualType withExactLocalFastQualifiers(unsigned TQs) const {
    return withoutLocalFastQualifiers().withFastQualifiers(TQs);
  }

  // Removes fast qualifiers, but leaves any extended qualifiers in place.
  QualType withoutLocalFastQualifiers() const {
    QualType T = *this;
    T.removeLocalFastQualifiers();
    return T;
  }

  QualType getCanonicalType() const;

  /// Return this type with all of the instance-specific qualifiers
  /// removed, but without removing any qualifiers that may have been applied
  /// through typedefs.
  QualType getLocalUnqualifiedType() const { return QualType(getTypePtr(), 0); }

  /// Retrieve the unqualified variant of the given type,
  /// removing as little sugar as possible.
  ///
  /// This routine looks through various kinds of sugar to find the
  /// least-desugared type that is unqualified. For example, given:
  ///
  /// \code
  /// typedef int Integer;
  /// typedef const Integer CInteger;
  /// typedef CInteger DifferenceType;
  /// \endcode
  ///
  /// Executing \c getUnqualifiedType() on the type \c DifferenceType will
  /// desugar until we hit the type \c Integer, which has no qualifiers on it.
  ///
  /// The resulting type might still be qualified if it's sugar for an array
  /// type.  To strip qualifiers even from within a sugared array type, use
  /// ASTContext::getUnqualifiedArrayType.
  inline QualType getUnqualifiedType() const;

  /// Retrieve the unqualified variant of the given type, removing as little
  /// sugar as possible.
  ///
  /// Like getUnqualifiedType(), but also returns the set of
  /// qualifiers that were built up.
  ///
  /// The resulting type might still be qualified if it's sugar for an array
  /// type.  To strip qualifiers even from within a sugared array type, use
  /// ASTContext::getUnqualifiedArrayType.
  inline SplitQualType getSplitUnqualifiedType() const;

  /// Determine whether this type is more qualified than the other
  /// given type, requiring exact equality for non-CVR qualifiers.
  bool isMoreQualifiedThan(QualType Other) const;

  /// Determine whether this type is at least as qualified as the other
  /// given type, requiring exact equality for non-CVR qualifiers.
  bool isAtLeastAsQualifiedAs(QualType Other) const;

  QualType getNonReferenceType() const;

  /// Determine the type of a (typically non-lvalue) expression with the
  /// specified result type.
  ///
  /// This routine should be used for expressions for which the return type is
  /// explicitly specified (e.g., in a cast or call) and isn't necessarily
  /// an lvalue. It removes a top-level reference (since there are no
  /// expressions of reference type) and deletes top-level cvr-qualifiers
  /// from non-class types (in C++) or all types (in C).
  QualType getNonLValueExprType(const ASTContext &Context) const;

  /// Return the specified type with any "sugar" removed from
  /// the type.  This takes off typedefs, typeof's etc.  If the outer level of
  /// the type is already concrete, it returns it unmodified.  This is similar
  /// to getting the canonical type, but it doesn't remove *all* typedefs.  For
  /// example, it returns "T*" as "T*", (not as "int*"), because the pointer is
  /// concrete.
  ///
  /// Qualifiers are left in place.
  QualType getDesugaredType(const ASTContext &Context) const {
    return getDesugaredType(*this, Context);
  }

  SplitQualType getSplitDesugaredType() const {
    return getSplitDesugaredType(*this);
  }

  /// Return the specified type with one level of "sugar" removed from
  /// the type.
  ///
  /// This routine takes off the first typedef, typeof, etc. If the outer level
  /// of the type is already concrete, it returns it unmodified.
  QualType getSingleStepDesugaredType(const ASTContext &Context) const {
    return getSingleStepDesugaredTypeImpl(*this, Context);
  }

  /// Returns the specified type after dropping any
  /// outer-level parentheses.
  QualType IgnoreParens() const;

  /// Indicate whether the specified types and qualifiers are identical.
  friend bool operator==(const QualType &LHS, const QualType &RHS) {
    return LHS.Value == RHS.Value;
  }
  friend bool operator!=(const QualType &LHS, const QualType &RHS) {
    return LHS.Value != RHS.Value;
  }
  friend bool operator<(const QualType &LHS, const QualType &RHS) {
    return LHS.Value < RHS.Value;
  }

  static std::string getAsString(SplitQualType split,
                                 const PrintingPolicy &Policy) {
    return getAsString(split.Ty, split.Quals, Policy);
  }
  static std::string getAsString(const Type *ty, Qualifiers qs,
                                 const PrintingPolicy &Policy);

  std::string getAsString() const;
  std::string getAsString(const PrintingPolicy &Policy) const;

  void print(raw_ostream &OS, const PrintingPolicy &Policy,
             const Twine &PlaceHolder = Twine(),
             unsigned Indentation = 0) const;

  static void print(SplitQualType split, raw_ostream &OS,
                    const PrintingPolicy &policy, const Twine &PlaceHolder,
                    unsigned Indentation = 0) {
    return print(split.Ty, split.Quals, OS, policy, PlaceHolder, Indentation);
  }

  static void print(const Type *ty, Qualifiers qs,
                    raw_ostream &OS, const PrintingPolicy &policy,
                    const Twine &PlaceHolder,
                    unsigned Indentation = 0);

  void getAsStringInternal(std::string &Str,
                           const PrintingPolicy &Policy) const;

  static void getAsStringInternal(SplitQualType split, std::string &out,
                                  const PrintingPolicy &policy) {
    return getAsStringInternal(split.Ty, split.Quals, out, policy);
  }

  static void getAsStringInternal(const Type *ty, Qualifiers qs,
                                  std::string &out,
                                  const PrintingPolicy &policy);

  class StreamedQualTypeHelper {
    const QualType &T;
    const PrintingPolicy &Policy;
    const Twine &PlaceHolder;
    unsigned Indentation;

  public:
    StreamedQualTypeHelper(const QualType &T, const PrintingPolicy &Policy,
                           const Twine &PlaceHolder, unsigned Indentation)
        : T(T), Policy(Policy), PlaceHolder(PlaceHolder),
          Indentation(Indentation) {}

    friend raw_ostream &operator<<(raw_ostream &OS,
                                   const StreamedQualTypeHelper &SQT) {
      SQT.T.print(OS, SQT.Policy, SQT.PlaceHolder, SQT.Indentation);
      return OS;
    }
  };

  StreamedQualTypeHelper stream(const PrintingPolicy &Policy,
                                const Twine &PlaceHolder = Twine(),
                                unsigned Indentation = 0) const {
    return StreamedQualTypeHelper(*this, Policy, PlaceHolder, Indentation);
  }

  void dump(const char *s) const;
  void dump() const;
  void dump(llvm::raw_ostream &OS) const;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(getAsOpaquePtr());
  }

  /// Check if this type has any address space qualifier.
  inline bool hasAddressSpace() const;

  /// Return the address space of this type.
  inline LangAS getAddressSpace() const;

  /// Returns gc attribute of this type.
  inline Qualifiers::GC getObjCGCAttr() const;

  /// true when Type is objc's weak.
  bool isObjCGCWeak() const {
    return getObjCGCAttr() == Qualifiers::Weak;
  }

  /// true when Type is objc's strong.
  bool isObjCGCStrong() const {
    return getObjCGCAttr() == Qualifiers::Strong;
  }

  /// Returns lifetime attribute of this type.
  Qualifiers::ObjCLifetime getObjCLifetime() const {
    return getQualifiers().getObjCLifetime();
  }

  bool hasNonTrivialObjCLifetime() const {
    return getQualifiers().hasNonTrivialObjCLifetime();
  }

  bool hasStrongOrWeakObjCLifetime() const {
    return getQualifiers().hasStrongOrWeakObjCLifetime();
  }

  // true when Type is objc's weak and weak is enabled but ARC isn't.
  bool isNonWeakInMRRWithObjCWeak(const ASTContext &Context) const;

  enum PrimitiveDefaultInitializeKind {
    /// The type does not fall into any of the following categories. Note that
    /// this case is zero-valued so that values of this enum can be used as a
    /// boolean condition for non-triviality.
    PDIK_Trivial,

    /// The type is an Objective-C retainable pointer type that is qualified
    /// with the ARC __strong qualifier.
    PDIK_ARCStrong,

    /// The type is an Objective-C retainable pointer type that is qualified
    /// with the ARC __weak qualifier.
    PDIK_ARCWeak,

    /// The type is a struct containing a field whose type is not PCK_Trivial.
    PDIK_Struct
  };

  /// Functions to query basic properties of non-trivial C struct types.

  /// Check if this is a non-trivial type that would cause a C struct
  /// transitively containing this type to be non-trivial to default initialize
  /// and return the kind.
  PrimitiveDefaultInitializeKind
  isNonTrivialToPrimitiveDefaultInitialize() const;

  enum PrimitiveCopyKind {
    /// The type does not fall into any of the following categories. Note that
    /// this case is zero-valued so that values of this enum can be used as a
    /// boolean condition for non-triviality.
    PCK_Trivial,

    /// The type would be trivial except that it is volatile-qualified. Types
    /// that fall into one of the other non-trivial cases may additionally be
    /// volatile-qualified.
    PCK_VolatileTrivial,

    /// The type is an Objective-C retainable pointer type that is qualified
    /// with the ARC __strong qualifier.
    PCK_ARCStrong,

    /// The type is an Objective-C retainable pointer type that is qualified
    /// with the ARC __weak qualifier.
    PCK_ARCWeak,

    /// The type is a struct containing a field whose type is neither
    /// PCK_Trivial nor PCK_VolatileTrivial.
    /// Note that a C++ struct type does not necessarily match this; C++ copying
    /// semantics are too complex to express here, in part because they depend
    /// on the exact constructor or assignment operator that is chosen by
    /// overload resolution to do the copy.
    PCK_Struct
  };

  /// Check if this is a non-trivial type that would cause a C struct
  /// transitively containing this type to be non-trivial to copy and return the
  /// kind.
  PrimitiveCopyKind isNonTrivialToPrimitiveCopy() const;

  /// Check if this is a non-trivial type that would cause a C struct
  /// transitively containing this type to be non-trivial to destructively
  /// move and return the kind. Destructive move in this context is a C++-style
  /// move in which the source object is placed in a valid but unspecified state
  /// after it is moved, as opposed to a truly destructive move in which the
  /// source object is placed in an uninitialized state.
  PrimitiveCopyKind isNonTrivialToPrimitiveDestructiveMove() const;

  enum DestructionKind {
    DK_none,
    DK_cxx_destructor,
    DK_objc_strong_lifetime,
    DK_objc_weak_lifetime,
    DK_nontrivial_c_struct
  };

  /// Returns a nonzero value if objects of this type require
  /// non-trivial work to clean up after.  Non-zero because it's
  /// conceivable that qualifiers (objc_gc(weak)?) could make
  /// something require destruction.
  DestructionKind isDestructedType() const {
    return isDestructedTypeImpl(*this);
  }

  /// Check if this is or contains a C union that is non-trivial to
  /// default-initialize, which is a union that has a member that is non-trivial
  /// to default-initialize. If this returns true,
  /// isNonTrivialToPrimitiveDefaultInitialize returns PDIK_Struct.
  bool hasNonTrivialToPrimitiveDefaultInitializeCUnion() const;

  /// Check if this is or contains a C union that is non-trivial to destruct,
  /// which is a union that has a member that is non-trivial to destruct. If
  /// this returns true, isDestructedType returns DK_nontrivial_c_struct.
  bool hasNonTrivialToPrimitiveDestructCUnion() const;

  /// Check if this is or contains a C union that is non-trivial to copy, which
  /// is a union that has a member that is non-trivial to copy. If this returns
  /// true, isNonTrivialToPrimitiveCopy returns PCK_Struct.
  bool hasNonTrivialToPrimitiveCopyCUnion() const;

  /// Determine whether expressions of the given type are forbidden
  /// from being lvalues in C.
  ///
  /// The expression types that are forbidden to be lvalues are:
  ///   - 'void', but not qualified void
  ///   - function types
  ///
  /// The exact rule here is C99 6.3.2.1:
  ///   An lvalue is an expression with an object type or an incomplete
  ///   type other than void.
  bool isCForbiddenLValueType() const;

  /// Substitute type arguments for the Objective-C type parameters used in the
  /// subject type.
  ///
  /// \param ctx ASTContext in which the type exists.
  ///
  /// \param typeArgs The type arguments that will be substituted for the
  /// Objective-C type parameters in the subject type, which are generally
  /// computed via \c Type::getObjCSubstitutions. If empty, the type
  /// parameters will be replaced with their bounds or id/Class, as appropriate
  /// for the context.
  ///
  /// \param context The context in which the subject type was written.
  ///
  /// \returns the resulting type.
  QualType substObjCTypeArgs(ASTContext &ctx,
                             ArrayRef<QualType> typeArgs,
                             ObjCSubstitutionContext context) const;

  /// Substitute type arguments from an object type for the Objective-C type
  /// parameters used in the subject type.
  ///
  /// This operation combines the computation of type arguments for
  /// substitution (\c Type::getObjCSubstitutions) with the actual process of
  /// substitution (\c QualType::substObjCTypeArgs) for the convenience of
  /// callers that need to perform a single substitution in isolation.
  ///
  /// \param objectType The type of the object whose member type we're
  /// substituting into. For example, this might be the receiver of a message
  /// or the base of a property access.
  ///
  /// \param dc The declaration context from which the subject type was
  /// retrieved, which indicates (for example) which type parameters should
  /// be substituted.
  ///
  /// \param context The context in which the subject type was written.
  ///
  /// \returns the subject type after replacing all of the Objective-C type
  /// parameters with their corresponding arguments.
  QualType substObjCMemberType(QualType objectType,
                               const DeclContext *dc,
                               ObjCSubstitutionContext context) const;

  /// Strip Objective-C "__kindof" types from the given type.
  QualType stripObjCKindOfType(const ASTContext &ctx) const;

  /// Remove all qualifiers including _Atomic.
  QualType getAtomicUnqualifiedType() const;

private:
  // These methods are implemented in a separate translation unit;
  // "static"-ize them to avoid creating temporary QualTypes in the
  // caller.
  static bool isConstant(QualType T, const ASTContext& Ctx);
  static QualType getDesugaredType(QualType T, const ASTContext &Context);
  static SplitQualType getSplitDesugaredType(QualType T);
  static SplitQualType getSplitUnqualifiedTypeImpl(QualType type);
  static QualType getSingleStepDesugaredTypeImpl(QualType type,
                                                 const ASTContext &C);
  static DestructionKind isDestructedTypeImpl(QualType type);
  static bool isCanonicalAsParamImpl(const Type *Ty);
  static QualType getCanonicalTypeFromType(const Type *Ty);
  static bool isAtLeastAsQualifiedAs(QualType self, QualType other);
};

} // namespace clang

namespace llvm {

/// Implement simplify_type for QualType, so that we can dyn_cast from QualType
/// to a specific Type class.
template<> struct simplify_type< ::clang::QualType> {
  using SimpleType = const ::clang::Type *;

  static SimpleType getSimplifiedValue(::clang::QualType Val) {
    return Val.getTypePtr();
  }
};

// Teach SmallPtrSet that QualType is "basically a pointer".
template<>
struct PointerLikeTypeTraits<clang::QualType> {
  static inline void *getAsVoidPointer(clang::QualType P) {
    return P.getAsOpaquePtr();
  }

  static inline clang::QualType getFromVoidPointer(void *P) {
    return clang::QualType::getFromOpaquePtr(P);
  }

  // Various qualifiers go in low bits.
  static constexpr int NumLowBitsAvailable = 0;
};

} // namespace llvm

namespace clang {

/// Base class that is common to both the \c ExtQuals and \c Type
/// classes, which allows \c QualType to access the common fields between the
/// two.
class ExtQualsTypeCommonBase {
  friend class ExtQuals;
  friend class QualType;
  friend class Type;

  /// The "base" type of an extended qualifiers type (\c ExtQuals) or
  /// a self-referential pointer (for \c Type).
  ///
  /// This pointer allows an efficient mapping from a QualType to its
  /// underlying type pointer.
  const Type *const BaseType;

  /// The canonical type of this type.  A QualType.
  QualType CanonicalType;

  ExtQualsTypeCommonBase(const Type *baseType, QualType canon)
      : BaseType(baseType), CanonicalType(canon) {}
};

/// We can encode up to four bits in the low bits of a
/// type pointer, but there are many more type qualifiers that we want
/// to be able to apply to an arbitrary type.  Therefore we have this
/// struct, intended to be heap-allocated and used by QualType to
/// store qualifiers.
///
/// The current design tags the 'const', 'restrict', and 'volatile' qualifiers
/// in three low bits on the QualType pointer; a fourth bit records whether
/// the pointer is an ExtQuals node. The extended qualifiers (address spaces,
/// Objective-C GC attributes) are much more rare.
class ExtQuals : public ExtQualsTypeCommonBase, public llvm::FoldingSetNode {
  // NOTE: changing the fast qualifiers should be straightforward as
  // long as you don't make 'const' non-fast.
  // 1. Qualifiers:
  //    a) Modify the bitmasks (Qualifiers::TQ and DeclSpec::TQ).
  //       Fast qualifiers must occupy the low-order bits.
  //    b) Update Qualifiers::FastWidth and FastMask.
  // 2. QualType:
  //    a) Update is{Volatile,Restrict}Qualified(), defined inline.
  //    b) Update remove{Volatile,Restrict}, defined near the end of
  //       this header.
  // 3. ASTContext:
  //    a) Update get{Volatile,Restrict}Type.

  /// The immutable set of qualifiers applied by this node. Always contains
  /// extended qualifiers.
  Qualifiers Quals;

  ExtQuals *this_() { return this; }

public:
  ExtQuals(const Type *baseType, QualType canon, Qualifiers quals)
      : ExtQualsTypeCommonBase(baseType,
                               canon.isNull() ? QualType(this_(), 0) : canon),
        Quals(quals) {
    assert(Quals.hasNonFastQualifiers()
           && "ExtQuals created with no fast qualifiers");
    assert(!Quals.hasFastQualifiers()
           && "ExtQuals created with fast qualifiers");
  }

  Qualifiers getQualifiers() const { return Quals; }

  bool hasObjCGCAttr() const { return Quals.hasObjCGCAttr(); }
  Qualifiers::GC getObjCGCAttr() const { return Quals.getObjCGCAttr(); }

  bool hasObjCLifetime() const { return Quals.hasObjCLifetime(); }
  Qualifiers::ObjCLifetime getObjCLifetime() const {
    return Quals.getObjCLifetime();
  }

  bool hasAddressSpace() const { return Quals.hasAddressSpace(); }
  LangAS getAddressSpace() const { return Quals.getAddressSpace(); }

  const Type *getBaseType() const { return BaseType; }

public:
  void Profile(llvm::FoldingSetNodeID &ID) const {
    Profile(ID, getBaseType(), Quals);
  }

  static void Profile(llvm::FoldingSetNodeID &ID,
                      const Type *BaseType,
                      Qualifiers Quals) {
    assert(!Quals.hasFastQualifiers() && "fast qualifiers in ExtQuals hash!");
    ID.AddPointer(BaseType);
    Quals.Profile(ID);
  }
};

// Inline function definitions.

inline const Type *QualType::getTypePtr() const {
  return getCommonPtr()->BaseType;
}

inline const Type *QualType::getTypePtrOrNull() const {
  return (isNull() ? nullptr : getCommonPtr()->BaseType);
}

inline SplitQualType QualType::split() const {
  if (!hasLocalNonFastQualifiers())
    return SplitQualType(getTypePtrUnsafe(),
                         Qualifiers::fromFastMask(getLocalFastQualifiers()));

  const ExtQuals *eq = getExtQualsUnsafe();
  Qualifiers qs = eq->getQualifiers();
  qs.addFastQualifiers(getLocalFastQualifiers());
  return SplitQualType(eq->getBaseType(), qs);
}

inline Qualifiers QualType::getLocalQualifiers() const {
  Qualifiers Quals;
  if (hasLocalNonFastQualifiers())
    Quals = getExtQualsUnsafe()->getQualifiers();
  Quals.addFastQualifiers(getLocalFastQualifiers());
  return Quals;
}

inline Qualifiers QualType::getQualifiers() const {
  Qualifiers quals = getCommonPtr()->CanonicalType.getLocalQualifiers();
  quals.addFastQualifiers(getLocalFastQualifiers());
  return quals;
}

inline unsigned QualType::getCVRQualifiers() const {
  unsigned cvr = getCommonPtr()->CanonicalType.getLocalCVRQualifiers();
  cvr |= getLocalCVRQualifiers();
  return cvr;
}

inline QualType QualType::getCanonicalType() const {
  QualType canon = getCommonPtr()->CanonicalType;
  return canon.withFastQualifiers(getLocalFastQualifiers());
}

inline bool QualType::isCanonical() const {
  const Type *T = getTypePtr();
  return getCanonicalTypeFromType(T) == QualType(T, 0);
}

inline bool QualType::isCanonicalAsParam() const {
  if (!isCanonical()) return false;
  if (hasLocalQualifiers()) return false;
  return isCanonicalAsParamImpl(getTypePtr());
}

inline bool QualType::isConstQualified() const {
  return isLocalConstQualified() ||
         getCommonPtr()->CanonicalType.isLocalConstQualified();
}

inline bool QualType::isRestrictQualified() const {
  return isLocalRestrictQualified() ||
         getCommonPtr()->CanonicalType.isLocalRestrictQualified();
}


inline bool QualType::isVolatileQualified() const {
  return isLocalVolatileQualified() ||
         getCommonPtr()->CanonicalType.isLocalVolatileQualified();
}

inline bool QualType::hasQualifiers() const {
  return hasLocalQualifiers() ||
         getCommonPtr()->CanonicalType.hasLocalQualifiers();
}

/// Without requiring Type to be complete, get its canonical type.
inline QualType QualType::getCanonicalTypeFromType(const Type *Ty) {
  // Type is known to derive from ExtQualsTypeCommonBase, so this
  // reinterpret_cast is safe. It is necessary to avoid reqiring Type to be
  // complete.
  return reinterpret_cast<const ExtQualsTypeCommonBase *>(Ty)->CanonicalType;
}

inline QualType QualType::getUnqualifiedType() const {
  const Type *Ty = getTypePtr();
  if (!getCanonicalTypeFromType(Ty).hasLocalQualifiers())
    return QualType(Ty, 0);

  return QualType(getSplitUnqualifiedTypeImpl(*this).Ty, 0);
}

inline SplitQualType QualType::getSplitUnqualifiedType() const {
  if (!getCanonicalTypeFromType(getTypePtr()).hasLocalQualifiers())
    return split();

  return getSplitUnqualifiedTypeImpl(*this);
}

inline void QualType::removeLocalConst() {
  removeLocalFastQualifiers(Qualifiers::Const);
}

inline void QualType::removeLocalRestrict() {
  removeLocalFastQualifiers(Qualifiers::Restrict);
}

inline void QualType::removeLocalVolatile() {
  removeLocalFastQualifiers(Qualifiers::Volatile);
}

inline void QualType::removeLocalCVRQualifiers(unsigned Mask) {
  assert(!(Mask & ~Qualifiers::CVRMask) && "mask has non-CVR bits");
  static_assert((int)Qualifiers::CVRMask == (int)Qualifiers::FastMask,
                "Fast bits differ from CVR bits!");

  // Fast path: we don't need to touch the slow qualifiers.
  removeLocalFastQualifiers(Mask);
}

/// Check if this type has any address space qualifier.
inline bool QualType::hasAddressSpace() const {
  return getQualifiers().hasAddressSpace();
}

/// Return the address space of this type.
inline LangAS QualType::getAddressSpace() const {
  return getQualifiers().getAddressSpace();
}

/// Return the gc attribute of this type.
inline Qualifiers::GC QualType::getObjCGCAttr() const {
  return getQualifiers().getObjCGCAttr();
}

/// Determine whether this type is more
/// qualified than the Other type. For example, "const volatile int"
/// is more qualified than "const int", "volatile int", and
/// "int". However, it is not more qualified than "const volatile
/// int".
inline bool QualType::isMoreQualifiedThan(QualType other) const {
  Qualifiers MyQuals = getQualifiers();
  Qualifiers OtherQuals = other.getQualifiers();
  return (MyQuals != OtherQuals && MyQuals.compatiblyIncludes(OtherQuals));
}

/// Determine whether this type is at least
/// as qualified as the Other type. For example, "const volatile
/// int" is at least as qualified as "const int", "volatile int",
/// "int", and "const volatile int".
inline bool QualType::isAtLeastAsQualifiedAs(QualType other) const {
  return isAtLeastAsQualifiedAs(*this, other);
}

/// Insertion operator for diagnostics. This allows sending address spaces into
/// a diagnostic with <<.
inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           LangAS AS) {
  DB.AddTaggedVal(static_cast<std::underlying_type_t<LangAS>>(AS),
                  DiagnosticsEngine::ArgumentKind::ak_addrspace);
  return DB;
}

/// Insertion operator for partial diagnostics. This allows sending adress
/// spaces into a diagnostic with <<.
inline const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                           LangAS AS) {
  PD.AddTaggedVal(static_cast<std::underlying_type_t<LangAS>>(AS),
                  DiagnosticsEngine::ArgumentKind::ak_addrspace);
  return PD;
}

/// Insertion operator for diagnostics. This allows sending Qualifiers into a
/// diagnostic with <<.
inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           Qualifiers Q) {
  DB.AddTaggedVal(Q.getAsOpaqueValue(),
                  DiagnosticsEngine::ArgumentKind::ak_qual);
  return DB;
}

/// Insertion operator for partial diagnostics. This allows sending Qualifiers
/// into a diagnostic with <<.
inline const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                           Qualifiers Q) {
  PD.AddTaggedVal(Q.getAsOpaqueValue(),
                  DiagnosticsEngine::ArgumentKind::ak_qual);
  return PD;
}

/// Insertion operator for diagnostics.  This allows sending QualType's into a
/// diagnostic with <<.
inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           QualType T) {
  DB.AddTaggedVal(reinterpret_cast<intptr_t>(T.getAsOpaquePtr()),
                  DiagnosticsEngine::ak_qualtype);
  return DB;
}

/// Insertion operator for partial diagnostics.  This allows sending QualType's
/// into a diagnostic with <<.
inline const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                           QualType T) {
  PD.AddTaggedVal(reinterpret_cast<intptr_t>(T.getAsOpaquePtr()),
                  DiagnosticsEngine::ak_qualtype);
  return PD;
}

} // namespace clang

#endif // LLVM_CLANG_AST_TYPE_H
